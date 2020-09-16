from numpy.testing._private.utils import verbose
import pandas as pd
import numpy as np
from scipy.stats import norm
import cplex

from fps_dfs_optimizer.src.optimizer import LineupOptimizer


class LineupGenerator:

    POSITION_COLS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

    def __init__(
        self, df, n_lineups_to_generate, n_lineups_to_optimize, batch_size, 
        var_multiple=0.5, drop_fraction=0.5, time_limit=1, 
        duplicates_lim=100, verbose=False):

        self.df = df
        self.n_players = len(self.df)
        self.n_lineups_to_generate = n_lineups_to_generate
        self.n_lineups_to_optimize = n_lineups_to_optimize
        self.batch_size = batch_size
        self.var_multiple = var_multiple
        self.drop_fraction = drop_fraction
        self.time_limit = time_limit
        self.duplicates_lim = duplicates_lim
        self.verbose = verbose
        self.df_lineups = pd.DataFrame(columns=self.POSITION_COLS)

    def generate(self):
        
        duplicates = 0
        n_lineups = self.df_lineups.shape[0]
        while (duplicates < self.duplicates_lim) & (n_lineups < self.n_lineups_to_generate):

            lineups_left = self.n_lineups_to_generate - n_lineups
            if lineups_left < self.batch_size:
                batch = lineups_left
            else:
                batch = self.batch_size
            
            keep_idx = np.random.choice(
                self.n_players, 
                size=round(self.n_players * (1 - self.drop_fraction)),
                replace=False)
            keep = np.where(self.df['max_exp'] > (1 - self.drop_fraction))[0]
            keep_idx = np.unique(np.append(keep_idx, keep))
            df = self.df.iloc[keep_idx]
            df['max_exp'] = df['max_exp'] * (1 + self.drop_fraction)
            df['min_exp'] = df['min_exp'] * (1 + self.drop_fraction)
            optimizer = LineupOptimizer(df, batch, order=False, time_limit=self.time_limit, verbose=self.verbose)
            optimizer.solve()
            lineups = optimizer.sort_lineups()
            self.df_lineups = pd.concat((self.df_lineups, lineups))
            length = len(self.df_lineups)
            self.df_lineups.drop_duplicates(inplace=True)
            n_lineups = len(self.df_lineups)
            duplicates += (length - n_lineups)
            print('{} Lineups'.format(n_lineups))
            print('{} Duplicates'.format(duplicates))

        self.df_lineups.reset_index(inplace=True, drop=True)
        return self.df_lineups

    def get_player_distribution(self, df):
        flat = df.loc[:, self.POSITION_COLS].values.flatten()
        return pd.Series(flat).value_counts() / len(df)

    def get_lineup_score_dists(self, sims=10000):
        self.df_lineups['mean'] = 0
        self.df_lineups['std'] = 0

        self.df.index = self.df.display
        for idx in self.df_lineups.index:
            players = self.df_lineups.loc[idx, self.POSITION_COLS].values
            mean = self.df.loc[players, 'projections'].sum()
            dists = norm().rvs(size=(sims, len(players)))
            spread = dists @ np.expand_dims(self.df.loc[players, 'std'].values, axis=1)
            std = np.std(spread)
            self.df_lineups.loc[idx, 'mean'] = mean
            self.df_lineups.loc[idx, 'std'] = std

        return self.df_lineups

    def enforce_exposures(self):
        player_dist = self.get_player_distribution(self.df_lineups)
        players = list(player_dist.index)
        lineup_mtx = self._get_lineup_mtx(players)
        iterations = int(np.ceil(self.n_lineups_to_generate / 1000))
        lineup_mtx = pd.concat((lineup_mtx.T, self.df_lineups[['mean', 'std']]), axis=1)
        self.results = []
        for i in range(iterations):
            exp = ExposureEnforcer(
                lineup_mtx.iloc[1000 * i: 1000 * (i+1), :], 
                1000 / iterations, self.df.loc[players], self.var_multiple)
            self.results += exp.solve()

        idx = np.where(np.array(self.results).astype(int))[0]
        exp = ExposureEnforcer(
            lineup_mtx.iloc[idx, :], 
            self.n_lineups_to_optimize, self.df.loc[players], self.var_multiple)
        self.result = exp.solve()
        return self.df_lineups.loc[np.where(np.array(self.result).astype(int))[0]]

    def _get_lineup_mtx(self, players=None):
        if players is None:
            players = self.df.display.values

        lineup_mtx = np.zeros((len(players), self.df_lineups.shape[0]))
        lineups = self.df_lineups.loc[:, self.POSITION_COLS].values
        for i, player in enumerate(players):
            locs = np.where(np.array([player == p for p in lineups]))[0]
            lineup_mtx[i, locs] = np.ones((locs.shape[0]))
        
        return pd.DataFrame(lineup_mtx.astype(int), columns=list(self.df_lineups.index), index=players)

class ExposureEnforcer:

    def __init__(self, lineup_mtx, n_lineups_to_optimize, df, var_multiple, time_limit=60, verbose=True):
        self.lineup_mtx = lineup_mtx
        self.n_lineups_to_optimize = n_lineups_to_optimize
        self.df = df
        self.var_multiple = var_multiple
        self.time_limit = time_limit
        self.verbose = verbose

        self._provision_constraints_from_df()
        self._construct_lp()

    def _provision_constraints_from_df(self):
        self.min_exp = self.df['min_exp'].values
        self.max_exp = self.df['max_exp'].values
        self.mean = self.lineup_mtx['mean'].values
        self.std = self.lineup_mtx['std'].values
        self.n_lineups = self.lineup_mtx.shape[0]
        self.n_players = len(self.df)
        self.players = list(self.df.index)

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
        for i in range(self.n_lineups):
            self.lp.variables.add(
                names= ["x_" + str(i)])
            self.lp.variables.set_types("x_" + str(i), self.lp.variables.type.binary)
            self.lp.objective.set_linear(
                "x_" + str(i), 
                self.mean[i] + self.var_multiple * self.std[i])
        self.lp.objective.set_sense(self.lp.objective.sense.maximize)

    def _get_constraints(self):
        index = ["x_" + str(i) for i in range(self.n_lineups)]
        for j in range(self.n_players):
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=self.lineup_mtx.loc[:, self.players[j]] * 1.0)],
                rhs=[1.0 * np.floor(self.min_exp[j] * self.n_lineups_to_optimize)],
                names=["exp_limit_low_"+str(j)],
                senses=["G"])
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=self.lineup_mtx.loc[:, self.players[j]] * 1.0)],
                rhs=[1.0 * np.ceil(self.max_exp[j] * self.n_lineups_to_optimize)],
                names=["exp_limit_high_"+str(j)],
                senses=["L"])
        
        self.lp.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=index, 
                val=[1.0]*self.n_lineups)],
            rhs=[self.n_lineups_to_optimize],
            names=["Lineup limit"],
            senses=["L"])

    def solve(self):
        self.lp.solve()
        print(self.lp.solution.is_primal_feasible())
        if self.lp.solution.is_primal_feasible():
            self.result = self.lp.solution.get_values()
        else:
            self.result = "infeasible"

        return self.result
