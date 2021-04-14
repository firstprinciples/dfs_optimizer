import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cplex
import datetime as dt
import time
import copy
import pyomo.environ as pyo

class Exposures:

    def __init__(self, df, auto=False, auto_max=0.4, folder='../data/temp/'):
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
        self.need_exposures = False
        if not exposures:
            if self.auto:
                self._get_auto_exposures()
            else:
                self._get_exposures()
                self.need_exposures = True
        
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
        if not self.need_exposures:
            return self.df

        path = self.folder + self.filename + '.csv'
        self.df = pd.read_csv(path, index_col=0)
        self._get_datetime()
        return self.df

class LineupOptimizer:

    CAP = 50
    EXC_LIM = [3.0, 3.0, 3.0, 3.0, 2.0]
    PLAYERS = 8.0
    POSITION_COLS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

    def __init__(self, df, n_lineups, team_limits=None, order=False, time_limit=10.0, verbose=False):
        self.df = df
        self.n_lineups = n_lineups
        self._provision_team_limits(team_limits)
        self.order = order
        self.time_limit = time_limit
        self.verbose = verbose
        
        self._trim_players()
        self.n_players = len(self.df)

        self._provision_constraints_from_df()
        self._construct_lp()
        self.result = None
        self.df_sorted_lineups = pd.DataFrame(columns=self.POSITION_COLS)

    def _provision_team_limits(self, team_limits):
        if team_limits is None:
            self.team_limits = {}
        else:
            self.team_limits = team_limits
    
    def _provision_constraints_from_df(self):
        self.projections = self.df['projections'].values
        self.salaries = self.df['Salary'].values / 1000
        self.teams = self.df['TeamAbbrev'].values
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

        for team, lim in self.team_limits.items():
            for j in range(self.n_lineups):
                index = ["y_" + str(i) + '_' + str(j) for i in range(self.n_players) \
                    if self.teams[i]==team]
                self.lp.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=index, 
                        val=[1.0] * len(index))],
                    rhs=[float(lim)],
                    names=["team_limit"],
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
            n_games = len(self.df.loc[idx, 'Game'].unique())
            if n_games < 2:
                infeasibles += 1
                continue
            
            players = self.df.loc[idx, 'Name'].values
            tip_times = self.tip_times[idx]
            positions = np.concatenate((
                self.positions[idx], 
                np.expand_dims(self.guards[idx], axis=1), 
                np.expand_dims(self.forwards[idx], axis=1), 
                np.ones(
                    (int(self.PLAYERS), 1))), axis=1)
            sorter = LineupSorter(tip_times, positions)
            result = sorter.solve()
            if result != 'infeasible':
                result_inds = result.T @ np.expand_dims(np.arange(int(self.PLAYERS)), axis=1)
                sorted_lineups += [np.squeeze(players[result_inds.astype(int)])]
            else:
                infeasibles += 1
            
        if len(sorted_lineups) > 0:
            self.df_sorted_lineups = pd.DataFrame(np.stack(sorted_lineups), columns=self.POSITION_COLS)
            self.df_sorted_lineups.drop_duplicates(inplace=True)
            self.df_sorted_lineups.reset_index(0, inplace=True, drop=True)
            redundant = len(sorted_lineups) - len(self.df_sorted_lineups)
        else:
            self.df_sorted_lineups = pd.DataFrame()
            redundant = 0

        print('{} infeasible and {} redundant lineups dropped during sorting'.format(
            infeasibles, redundant
        ))
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
            index = ['y' + str(i) + str(j) for j in range(self.PLAYERS)]
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=[1.0] * self.PLAYERS)],
                rhs=[1.0],
                names=["player_limit" + str(i)],
                senses=["E"])
        for j in range(self.PLAYERS):
            index = ['y' + str(i) + str(j) for i in range(self.PLAYERS)]
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=[1.0] * self.PLAYERS)],
                rhs=[1.0],
                names=["position_limit"+str(j)],
                senses=["E"])
        for i in range(self.PLAYERS):
            for j in range(self.PLAYERS):
                self.lp.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=['y' + str(i) + str(j)], 
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


class Reoptimizer:
    
    def __init__(self, entries, cap_limit=50000, solver='glpk', executable=None,
                 timelimit=120, mipgap=0.001, verbose=False, visualize=False):
        self.entries = entries
        self.cap_limit = cap_limit
        self.solver = solver
        self.executable = executable
        self.timelimit = timelimit
        self.mipgap = mipgap
        self.verbose = verbose
        self.visualize = visualize

        self.df = self.entries.df
        self.df_lineups = self.entries.df_sheet_lineups
        self._get_locked_matrix()
        self._get_available_salary()
        self._get_unlocked_inds()
        self._create_model()

        self.fixed_last = 1200

    def initialize(self):
        self._get_model_data()
        self._create_instance()
        self._get_optimizer(use_time_lim=False)
        self.iterations = 0
        self.df_optimal = copy.deepcopy(self.df_locked)

    def solve(self, iter_lim=20):
        self.open_slots = (self.df_locked==0).sum().sum()
        print('Unfilled spots remaining: {}'.format(self.open_slots))
        if self.visualize:
            plt.rcParams['figure.figsize'] = 12, 8
            plt.imshow((self.df_locked==0).T)
            plt.show()
        
        while self.open_slots > 0:
            self._iterate_opt(iter_lim=iter_lim)
            self._get_optimizer()
            self.open_slots = (self.df_optimal==0).sum().sum()
            print('Unfilled spots remaining: {}'.format(self.open_slots))
            if self.visualize:
                plt.imshow((self.df_optimal==0).T)
                plt.show()

    def summarize(self):
        df_reopt_summary = pd.concat((
            self.entries.get_player_distribution(self.entries.df_sheet_lineups), 
            self.entries.get_player_distribution(self.df_optimal)), axis=1)
        df_reopt_summary.columns = ['Before', 'After']
        df_reopt_summary.fillna(0, inplace=True)
        df_reopt_summary['Difference'] = df_reopt_summary['After'] - df_reopt_summary['Before']
        df_reopt_summary['Magnitude'] = np.absolute(df_reopt_summary['Difference'])
        df_reopt_summary.sort_values(by='Magnitude', ascending=False, inplace=True)
        df_reopt_summary.drop('Magnitude', axis=1, inplace=True)
        return df_reopt_summary

    def _iterate_opt(self, iter_lim=20):
        self.opt.solve(self.instance, tee=self.verbose)
        self.iterations += 1
        cutoff = np.maximum(0.01, 1 - self.iterations * 0.05)
        self.binaries = 0
        if self.iterations > iter_lim:
            assignment_val = 0.5
        else:
            assignment_val = 1.0

        for n, p, l in self.npl_set:
            if self.instance.assignment_npl[(n, p, l)].value >= assignment_val:
                self.df_optimal.loc[int(l[2:]), p] = n
                self.instance.assignment_npl[(n, p, l)].domain = pyo.UnitInterval
                self.instance.assignment_npl[(n, p, l)].fix(1.0)
            elif self.instance.assignment_npl[(n, p, l)].value >= cutoff:
                self.instance.assignment_npl[(n, p, l)].domain = pyo.Binary
                self.binaries += 1
            else:
                self.instance.assignment_npl[(n, p, l)].fixed = False

        #if self.verbose:
        print('\n')
        print('Cutoff set to {}; {} variables converted to binaries'.format(cutoff, self.binaries))
        print('\n')

    def _get_optimizer(self, use_time_lim=True):
        if self.executable is None:
            self.opt = pyo.SolverFactory(self.solver)
        else:
            self.opt = pyo.SolverFactory(self.solver, executable=self.executable)

        if use_time_lim:
            if self.solver == 'glpk':
                self.opt.options["mipgap"] = self.mipgap
                self.opt.options["tmlim"] = self.timelimit
            elif self.solver == 'cbc':
                self.opt.options["sec"] = self.timelimit
            elif self.solver == 'cplex':
                self.opt.options["timelimit"] = self.timelimit

    def _get_locked_matrix(self):
        self.df_locked = pd.DataFrame(columns=self.entries.POSITION_COLS, index=self.df_lineups.index)
        if 'reopt' in self.df.columns:
            unlocked_lineups = []
            reopt_players = list(self.df.index[self.df['reopt']==1])
            for lineup in self.df_lineups.index:
                for col in self.entries.POSITION_COLS:
                    player = self.df_lineups.loc[lineup, col]
                    if player in reopt_players:
                        unlocked_lineups += [lineup]
                        break

            locked_lineups = set(self.df_lineups.index).difference(set(unlocked_lineups))
        else:
            locked_lineups = []
        
        for lineup in self.df_lineups.index:
            for col in self.entries.POSITION_COLS:
                player = self.df_lineups.loc[lineup, col]
                if self.df.loc[player, 'locked']:
                    self.df_locked.loc[lineup, col] = player
                elif lineup in locked_lineups:
                    self.df_locked.loc[lineup, col] = player
                else:
                    self.df_locked.loc[lineup, col] = 0

    def _get_available_salary(self):
        self.salary_available = {}
        for lineup in self.df_lineups.index:
            players = [self.df_locked.loc[lineup, j] for j in self.entries.POSITION_COLS]
            self.salary_available[lineup] = self.cap_limit - sum(self.df.loc[p, 'Salary'] for p in players if p != 0)

    def _get_model_data(self):
        self._get_unlocked_inds()
        self._get_model_sets()
        self._get_model_multisets()
        self._get_model_params()
        self._get_current_lineup_summary()

    def _get_unlocked_inds(self):
        self.unlocked = self.df.index[~self.df.locked]

    def _get_model_sets(self):
        self.name_set = self.unlocked.values.tolist()
        self.lineup_set = ['L_' + str(i) for i in self.df_lineups.index]
        self.position_set = self.entries.POSITION_COLS

    def _get_model_multisets(self):
        self.np_set = []
        for n in self.name_set:
            self.np_set += [(n, 'UTIL')]
            pos_split = self.df.loc[n, 'Position'].split('/')
            for p in pos_split:
                self.np_set += [(n, p)]
                if ('G' in p) & ((n, 'G') not in self.np_set):
                    self.np_set += [(n, 'G')]

                if ('F' in p) & ((n, 'F') not in self.np_set):
                    self.np_set += [(n, 'F')]

        self.pl_set = []
        for l in self.lineup_set:
            for p in self.position_set:
                if self.df_locked.loc[int(l[2:]), p] == 0:
                    self.pl_set += [(p, l)]

        self.npl_set = []
        for n in self.name_set:
            for l in self.lineup_set:
                for p in self.position_set:
                    if ((p, l) in self.pl_set) & ((n, p) in self.np_set):
                        self.npl_set += [(n, p, l)]

    def _get_model_params(self):
        # start times
        time_start = pd.Series([t.hour * 60 + t.minute + np.random.choice(5) \
            for t in self.df.Time[self.unlocked]], index=self.unlocked)
        time_start_norm = (time_start - time_start.min()) / (time_start.max() - time_start.min())
        self.start_times_n = time_start_norm.to_dict()

        # projections
        self.projections_n = self.df.projections[self.unlocked].to_dict()

        # salaries
        self.salaries_n = (self.df.Salary[self.unlocked] / 1000).to_dict()

        # exposures
        self.max_exp_n = (len(self.lineup_set) * self.df.max_exp[self.unlocked]).to_dict()
        self.min_exp_n = (len(self.lineup_set) * self.df.min_exp[self.unlocked]).to_dict()

        # available cap
        self.cap_l = {'L_'+str(i): self.salary_available[i]/1000 \
                for i in self.salary_available.keys()}

        # spots open in lineup
        self.players_per_lineup_l = pd.Series(np.where(self.df_locked==0)[0]).value_counts()
        for i in range(len(self.lineup_set)):
            if i not in self.players_per_lineup_l.index:
                self.players_per_lineup_l[i] = 0

        self.players_per_lineup_l = {'L_' + str(i): float(self.players_per_lineup_l.loc[i]) \
            for i in self.players_per_lineup_l.index}

        # start time value
        self.start_time_bonus_p = {p: 0.0 for p in self.position_set[:5]}
        self.start_time_bonus_p['G'] = 0.1
        self.start_time_bonus_p['F'] = 0.1
        self.start_time_bonus_p['UTIL'] = 0.2

    def _get_current_lineup_summary(self):
        self.df_current = pd.concat(
            (pd.Series(self.players_per_lineup_l), pd.Series(self.cap_l)), axis=1)
        self.df_current.columns = ['players', 'cap']
        self.df_current.sort_values(by='cap')

    def _create_instance(self):
        data = {
            None: {
                'name_set': {None: self.name_set},
                'lineup_set': {None: self.lineup_set},
                'position_set': {None: self.position_set},
                'np_set': self.np_set,
                'pl_set': self.pl_set,
                'npl_set': self.npl_set,
                'projections_n': self.projections_n,
                'salaries_n': self.salaries_n,
                'cap_l': self.cap_l,
                'players_per_lineup_l': self.players_per_lineup_l,
                'max_exp_n': self.max_exp_n,
                'min_exp_n': self.min_exp_n,
                'start_times_n': self.start_times_n,
                'start_time_bonus_p': self.start_time_bonus_p}}

        self.instance = self.model.create_instance(data)

    def _create_model(self):
        self.model = pyo.AbstractModel()
        self._get_base_sets()
        self._get_multisets()
        self._get_params()
        self._get_vars()
        self._get_constraints()
        self._get_objective()

    def _get_base_sets(self):
        self.model.name_set = pyo.Set()
        self.model.lineup_set = pyo.Set()
        self.model.position_set = pyo.Set()

    def _get_multisets(self):
        self.model.nl_set = self.model.name_set * self.model.lineup_set
        self.model.np_set = pyo.Set(within = self.model.name_set * self.model.position_set)
        self.model.pl_set = pyo.Set(within = self.model.position_set * self.model.lineup_set)
        self.model.npl_set = pyo.Set(within = self.model.name_set * self.model.position_set * self.model.lineup_set)

    def _get_params(self):
        self.model.projections_n = pyo.Param(self.model.name_set)
        self.model.salaries_n = pyo.Param(self.model.name_set)
        self.model.start_times_n = pyo.Param(self.model.name_set)
        self.model.start_time_bonus_p = pyo.Param(self.model.position_set)
        self.model.cap_l = pyo.Param(self.model.lineup_set)
        self.model.players_per_lineup_l = pyo.Param(self.model.lineup_set)
        self.model.max_exp_n = pyo.Param(self.model.name_set)
        self.model.min_exp_n = pyo.Param(self.model.name_set)

    def _get_vars(self):
        self.model.assignment_npl = pyo.Var(self.model.npl_set, within=pyo.UnitInterval)

    def _get_objective(self):
        self.model.obj = pyo.Objective(rule=self._objective_rule, sense=pyo.maximize)

    def _get_constraints(self):
        self.model.cap_limit_l = pyo.Constraint(self.model.lineup_set, rule=self._cap_limit_rule)
        self.model.player_limit_l = pyo.Constraint(self.model.lineup_set, rule=self._player_limit_rule)
        self.model.exposure_limit_n = pyo.Constraint(self.model.name_set, rule=self._exposure_limit_rule)
        self.model.fill_spot_pl = pyo.Constraint(self.model.pl_set, rule=self._position_fill_rule)
        self.model.player_in_lineup_once_nl = pyo.Constraint(self.model.nl_set, rule=self._player_per_lineup_rule)

    @staticmethod
    def _objective_rule(model):
        objective = sum(
            (model.projections_n[n] + model.start_times_n[n] * model.start_time_bonus_p[p]) \
                * model.assignment_npl[(n, p, l)] \
                for n, p, l in model.npl_set)
        return objective

    @staticmethod
    def _cap_limit_rule(model, l):
        salary = sum(model.salaries_n[n] * model.assignment_npl[(n, p, l)] \
            for n, p in model.np_set if (n, p, l) in model.npl_set)
        return (None, salary, model.cap_l[l])

    @staticmethod
    def _player_limit_rule(model, l):
        players = sum(model.assignment_npl[(n, p, l)] \
            for n, p in model.np_set if (n, p, l) in model.npl_set)
        return (model.players_per_lineup_l[l], players, model.players_per_lineup_l[l])

    @staticmethod
    def _exposure_limit_rule(model, n):
        exposure = sum(model.assignment_npl[(n, p, l)] \
            for p, l in model.pl_set if (n, p, l) in model.npl_set)
        return (model.min_exp_n[n], exposure, model.max_exp_n[n])

    @staticmethod
    def _position_fill_rule(model, p, l):
        spot_fill = sum(model.assignment_npl[(n, p, l)] \
            for n in model.name_set if (n, p, l) in model.npl_set)
        return (1.0, spot_fill, 1.0)

    @staticmethod
    def _player_per_lineup_rule(model, n, l):
        player_in_lineup = sum(model.assignment_npl[(n, p, l)] \
            for p in model.position_set if (n, p, l) in model.npl_set)
        return (0.0, player_in_lineup, 1.0)