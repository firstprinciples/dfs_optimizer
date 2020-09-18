import numpy as np
import pandas as pd
import cplex
import datetime as dt
import time

class Exposures:

    def __init__(
        self, df, auto=False, auto_max=0.4, folder='../data/temp/'):
        self.df = df
        self.auto = auto
        self.auto_max = auto_max
        self.folder = folder
        self.filename = dt.date.fromtimestamp(time.time()).strftime("%Y%m%d")
        self.cols_out = [
            'Name', 'ID', 'Position', 'Salary', 'Status', 
            'TeamAbbrev', 'Time', 'projections', 'std',
            'min_exp', 'max_exp']
        exposures = self._check_exposures()
        if not exposures:
            if self.auto:
                self._get_auto_exposures()
            else:
                self._get_exposures()
        
        self._get_datetime()

    def _get_auto_exposures(self):
        self.df['min_exp'] = 0
        self.df['max_exp'] = self.auto_max

    def _check_exposures(self):
        if ('min_exp' in self.df.columns) and ('max_exp' in self.df.columns):
            return True
        else:
            return False
        
    def _get_exposures(self):
        path = self.folder + self.filename + '.csv'
        self.df['min_exp'] = 0
        self.df['max_exp'] = self.auto_max
        self.df[self.cols_out].to_csv(path)
        print('Please fill out exposures in ' + path)

    def _get_datetime(self):
        self.df['Time'] = pd.to_datetime(self.df['Time'])

    def read_exposures(self):
        path = self.folder + self.filename + '.csv'
        self.df = pd.read_csv(path, index_col=0)
        self._get_datetime()
        return self.df

class LineupOptimizer:

    CAP = 50
    EXC_LIM = [3.0, 3.0, 3.0, 3.0, 2.0]
    PLAYERS = 8.0
    POSITION_COLS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

    def __init__(self, df, n_lineups, order=False, time_limit=10.0, verbose=False):
        self.df = df
        self.n_lineups = n_lineups
        self.order = order
        self.time_limit = time_limit
        self.verbose = verbose
        
        self._trim_players()
        self.n_players = len(self.df)

        self._provision_constraints_from_df()
        self._construct_lp()
        self.result = None
        self.df_sorted_lineups = pd.DataFrame(columns=self.POSITION_COLS)
    
    def _provision_constraints_from_df(self):
        self.projections = self.df['projections'].values
        self.salaries = self.df['Salary'].values / 1000
        pos_split = [pos.split('/') for pos in self.df['Position'].values]
        positions_mtx = np.zeros((len(pos_split), 5))
        for i in range(len(pos_split)):
            if 'PG' in pos_split[i]:
                positions_mtx[i,0] = 1
            if 'SG' in pos_split[i]:
                positions_mtx[i,1] = 1
            if 'SF' in pos_split[i]:
                positions_mtx[i,2] = 1
            if 'PF' in pos_split[i]:
                positions_mtx[i,3] = 1
            if 'C' in pos_split[i]:
                positions_mtx[i,4] = 1
        self.positions = positions_mtx
        self.exclusive = np.expand_dims(np.sum(positions_mtx, axis=1)==1 + 0.0, axis=1)
        self.exclusive = self.positions * self.exclusive
        self.guards = np.minimum(1, np.sum(positions_mtx[:, :2], axis=1))
        self.forwards = np.minimum(1, np.sum(positions_mtx[:, 2:4],axis=1))
        self.max_exp = self.df['max_exp'].values
        self.min_exp = self.df['min_exp'].values
        times = []
        for idx in self.df.index:
            start = self.df.loc[idx, 'Time'].to_pydatetime()
            times += [start.hour + start.minute / 60 + start.second / 3600]
        
        self.tip_times = np.array(times) + np.arange(self.n_players) / 3600

    def _trim_players(self):
        self.df = self.df[self.df['max_exp'] > 0]
        self.df['ppd'] = self.df['projections'] / self.df['Salary'] * 1000
        self.df.loc[self.df['min_exp'] > 0, 'ppd'] = np.max(self.df.ppd)
        self.df.sort_values(by='ppd', inplace=True, ascending=False)
        lim_players = 999 // self.n_lineups
        self.df = self.df.iloc[:lim_players]
        self.df.reset_index(0, inplace=True, drop=True)

    def _construct_lp(self):
        self.lp = cplex.Cplex()
        self.lp.parameters.timelimit.set(self.time_limit)
        self._get_variables_and_objective()
        self._get_constraints()
        if not self.verbose:
            self.lp.set_log_stream(None)
            self.lp.set_error_stream(None)
            self.lp.set_warning_stream(None)
            self.lp.set_results_stream(None)

    def _get_variables_and_objective(self):
        for i in range(self.n_players):
            for j in range(self.n_lineups):
                self.lp.variables.add(
                    names= ["y_" + str(i) + '_' + str(j)])
                self.lp.variables.set_types("y_" + str(i) + '_' + str(j), self.lp.variables.type.binary)
                self.lp.objective.set_linear(
                    "y_" + str(i) + '_' + str(j), 
                    (1 + self.order * (self.n_lineups - j - 1)) * self.projections[i])
        self.lp.objective.set_sense(self.lp.objective.sense.maximize)

    def _get_constraints(self):
        for j in range(self.n_lineups):
            index = ["y_" + str(i) + '_' + str(j) for i in range(self.n_players)]
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=self.salaries)],
                rhs=[self.CAP],
                names=["cap_limit"],
                senses=["L"])
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=[1.0] * self.n_players)],
                rhs=[self.PLAYERS],
                names=["player_limit"],
                senses=["E"])
            for k in range(5):
                self.lp.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=index, 
                        val=self.positions[:, k])],
                    rhs=[1.0],
                    names = ["position_limit"],
                    senses = ["G"])
                self.lp.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=index, 
                        val=self.exclusive[:, k])],
                    rhs=[self.EXC_LIM[k]],
                    names=["exc_limit"],
                    senses=["L"])

            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=self.guards)],
                rhs=[3.0],
                names=["guard_limit"],
                senses=["G"])
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=self.forwards)],
                rhs=[3.0],
                names=["forward_limit"],
                senses=["G"])

        for i in range(self.n_players):
            index = ["y_" + str(i) + '_' + str(j) for j in range(self.n_lineups)]
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=[1.0] * self.n_lineups)],
                rhs=[np.floor(self.min_exp[i] * self.n_lineups)],
                names=["exp_limit_low"],
                senses=["G"])
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=[1.0] * self.n_lineups)],
                rhs=[np.ceil(self.max_exp[i] * self.n_lineups)],
                names=["exp_limit_high"],
                senses=["L"])

    def solve(self):
        self.lp.solve()
        if self.lp.solution.is_primal_feasible():
            self.result = np.reshape(
                self.lp.solution.get_values(), (self.n_players, self.n_lineups))
            self.result = np.around(self.result)
        else:
            self.result = "infeasible"
        return self.result

    def sort_lineups(self):
        sorted_lineups = []
        infeasibles = 0
        for j in range(self.n_lineups):
            idx = np.where(self.result[:, j])[0]
            players = self.df.loc[idx, 'Name'].values
            tip_times = self.tip_times[idx]
            positions = np.concatenate((
                self.positions[idx], 
                np.expand_dims(self.guards[idx], axis=1), 
                np.expand_dims(self.forwards[idx], axis=1), 
                np.ones((int(self.PLAYERS), 1))
                ), axis=1)
            sorter = LineupSorter(tip_times, positions)
            result = sorter.solve()
            if result != 'infeasible':
                result_inds = result.T @ np.expand_dims(np.arange(int(self.PLAYERS)), axis=1)
                sorted_lineups += [np.squeeze(players[result_inds.astype(int)])]
            else:
                infeasibles += 1
            
        print(str(infeasibles) + ' Infeasible lineups dropped')
        if len(sorted_lineups) > 0:
            self.df_sorted_lineups = pd.DataFrame(np.stack(sorted_lineups), columns=self.POSITION_COLS)
            self.df_sorted_lineups.drop_duplicates(inplace=True)
            self.df_sorted_lineups.reset_index(0, inplace=True, drop=True)
        return self.df_sorted_lineups


class LineupSorter:

    PLAYERS = 8

    def __init__(self, tip_times, positions, time_limit=1, verbose=False):
        self.tip_times = tip_times
        self.positions = positions
        self.time_limit = time_limit
        self.verbose = verbose
        self._construct_lp()

    def _construct_lp(self):
        self.lp = cplex.Cplex()
        self.lp.parameters.timelimit.set(self.time_limit)
        self._get_variables_and_objective()
        self._get_constraints()
        if not self.verbose:
            self.lp.set_log_stream(None)
            self.lp.set_error_stream(None)
            self.lp.set_warning_stream(None)
            self.lp.set_results_stream(None)

    def _get_variables_and_objective(self):
        for i in range(self.PLAYERS):
            for j in range(self.PLAYERS):
                self.lp.variables.add(
                    names= ["y" + str(i) + str(j)])
                self.lp.variables.set_types("y" + str(i) + str(j), self.lp.variables.type.binary)
        for i in range(self.PLAYERS):
            for j in range(5):
                self.lp.objective.set_linear(
                    "y" + str(i) + str(j), 0.1 * j * self.tip_times[i])
            for j in range(5, self.PLAYERS-1):
                self.lp.objective.set_linear(
                    "y" + str(i) + str(j), self.tip_times[i])
            self.lp.objective.set_linear(
                "y" + str(i) + str(7), 2 * self.tip_times[i])
        self.lp.objective.set_sense(self.lp.objective.sense.maximize)

    def _get_constraints(self):
        for i in range(self.PLAYERS):
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=[i * self.PLAYERS + j for j in range(self.PLAYERS)], 
                    val=[1.0] * self.PLAYERS)],
                rhs=[1.0],
                names=["player_limit" + str(i)],
                senses=["E"])
        for j in range(self.PLAYERS):
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=[i * self.PLAYERS + j for i in range(self.PLAYERS)], 
                    val=[1.0] * self.PLAYERS)],
                rhs=[1.0],
                names=["position_limit"+str(j)],
                senses=["E"])
        for i in range(self.PLAYERS):
            for j in range(self.PLAYERS):
                self.lp.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i * self.PLAYERS + j], 
                        val=[1.0])],
                    rhs=[self.positions[i, j]],
                    names=["position_constraint"+str(i)+str(j)],
                    senses=["L"])

    def solve(self):
        self.lp.solve()
        if self.lp.solution.is_primal_feasible():
            self.result = np.reshape(
                self.lp.solution.get_values(), (self.PLAYERS, self.PLAYERS))
            self.result = np.around(self.result)
        else:
            self.result = "infeasible"
        return self.result