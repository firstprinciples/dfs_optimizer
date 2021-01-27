from fps_dfs_optimizer.src.draftkings_tools import get_players_from_salaries
import pandas as pd
import numpy as np
import time
from scipy.stats import norm
import cplex

from fps_dfs_optimizer.src.optimizer import LineupOptimizer


class LineupGenerator:

    POSITION_COLS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']

    def __init__(
        self, df, batch_size, 
        drop_fraction=0.5, time_limit=1, 
        duplicates_lim=100, verbose=False):

        self.df = df[df['max_exp'] > 0]
        self.n_players = len(self.df)
        self.batch_size = batch_size
        self.drop_fraction = drop_fraction
        self.time_limit = time_limit
        self.duplicates_lim = duplicates_lim
        self.verbose = verbose
        self.df_lineups = pd.DataFrame(columns=self.POSITION_COLS)

    def generate_n_lineups(self, n_lineups_to_generate):
        return self.generate(n_lineups_to_generate=n_lineups_to_generate)

    def generate_n_minutes(self, gen_time):
        return self.generate(gen_time=gen_time)

    def generate(self, n_lineups_to_generate=None, gen_time=None):
        if n_lineups_to_generate is None:
            n_lineups_to_generate = 10000
        
            if gen_time is None:
                gen_time = 5

        self.df_lineups = pd.concat((self.df_lineups, 
            self._generate(n_lineups_to_generate, gen_time)))     
        self.df_lineups.drop_duplicates(inplace=True)
        self.df_lineups.reset_index(inplace=True, drop=True)
        self._get_lineup_salaries()
        return self.df_lineups

    def update_exp(self, path):
        self.df = get_players_from_salaries(path)

    def _generate(self, n_lineups_to_generate, gen_time):
        if gen_time is None:
            gen_time = 60
        else:
            self.duplicates_lim = 1E6

        start_time = time.time()
        df_lineups = pd.DataFrame(columns=self.POSITION_COLS)
        duplicates = 0
        n_lineups = df_lineups.shape[0]
        run_time = time.time() - start_time
        while (duplicates < self.duplicates_lim) & (n_lineups < n_lineups_to_generate) & (run_time / 60 < gen_time):

            lineups_left = n_lineups_to_generate - n_lineups
            if lineups_left < self.batch_size:
                batch = lineups_left
            else:
                rand = np.random.choice(self.batch_size // 2)
                batch = rand + self.batch_size // 2
            
            keep_idx = np.random.choice(
                self.n_players, 
                size=round(self.n_players * (1 - self.drop_fraction)),
                replace=False)
            keep = np.where(self.df['max_exp'] > (1 - self.drop_fraction))[0]
            keep_idx = np.unique(np.append(keep_idx, keep))
            df = self.df.iloc[keep_idx]
            df['max_exp'] = df['max_exp'] * (1 + self.drop_fraction)
            df['min_exp'] = df['min_exp'] * (min(1, 1 + self.drop_fraction))
            optimizer = LineupOptimizer(
                df, batch, order=False, 
                time_limit=self.time_limit, verbose=self.verbose
            )
            optimizer.solve()
            if optimizer.result == 'infeasible':
                print('batch infeasible')
                continue

            lineups = optimizer.sort_lineups()
            df_lineups = pd.concat((df_lineups, lineups))
            length = len(df_lineups)
            df_lineups.drop_duplicates(inplace=True)
            n_lineups = len(df_lineups)
            duplicates += (length - n_lineups)
            run_time = time.time() - start_time
            print('\n')
            print('Elapsed time: {:0.2f} minutes\nLineups created: {}\nDuplicates removed: {}'.format(
                run_time/60, n_lineups, duplicates
            ))
            print('\n')

        df_lineups.reset_index(inplace=True, drop=True)
        return df_lineups

    def get_player_distribution(self, df):
        flat = df.loc[:, self.POSITION_COLS].values.flatten()
        return pd.Series(flat).value_counts() / len(df)

    def get_lineup_score_dists(self, sims=10000):
        self.df_lineups['mean'] = 0
        self.df_lineups['std'] = 0

        self.df.index = self.df.Name
        for idx in self.df_lineups.index:
            players = self.df_lineups.loc[idx, self.POSITION_COLS].values
            mean = self.df.loc[players, 'projections'].sum()
            dists = norm().rvs(size=(sims, len(players)))
            spread = dists @ np.expand_dims(self.df.loc[players, 'std'].values, axis=1)
            std = np.std(spread)
            self.df_lineups.loc[idx, 'mean'] = mean
            self.df_lineups.loc[idx, 'std'] = std
        
        return self.df_lineups

    def get_lineup_correlation(self, sims=10000):
        self.df.index = self.df.Name
        dists = norm().rvs(size=(len(self.df), sims))
        player_sims = self.df['projections'].values.reshape(-1, 1) + \
            dists * self.df['std'].values.reshape(-1, 1)
        df_sims = pd.DataFrame(player_sims, index=self.df.index)

        lineup_sims = {}
        for idx in self.df_lineups.index:
            players = self.df_lineups.loc[idx, self.POSITION_COLS].values
            lineup_sims[idx] = df_sims.loc[players, :].sum(axis=0)
            mean = lineup_sims[idx].mean()
            std = lineup_sims[idx].std()
            self.df_lineups.loc[idx, 'mean'] = mean
            self.df_lineups.loc[idx, 'std'] = std
        
        lineup_sims = pd.DataFrame(lineup_sims)
        lineup_sims_norm = lineup_sims - lineup_sims.mean(axis=0)
        self.lineup_cov = (lineup_sims_norm.T @ lineup_sims_norm) / sims + \
            0.001 * np.eye(len(self.df_lineups))
        return self.df_lineups

    def enforce_exposures(self, var_multiple, n_lineups_to_optimize, verbose=False):
        player_dist = self.get_player_distribution(self.df_lineups)
        players = list(player_dist.index)
        lineup_mtx = self._get_lineup_mtx(players)
        max_ = np.minimum(1000, len(self.df_lineups))
        iterations = int(np.ceil(len(self.df_lineups) / max_))
        lineup_mtx = pd.concat((lineup_mtx.T, self.df_lineups[['mean', 'std']]), axis=1)
        self.results = []
        if iterations > 1:
            frac_to_select = 1000 / ((iterations - 1) * 1000 + len(self.df_lineups) % 1000)
            frac_to_select = min(0.3, frac_to_select)
            for k in range(iterations):
                iter_lineupmtx = lineup_mtx.iloc[max_ * k: max_ * (k+1), :]
                n_to_select = frac_to_select * len(iter_lineupmtx)
                exp = ExposureEnforcer(
                    iter_lineupmtx, round(n_to_select), 
                    self.df.loc[players], var_multiple, verbose=verbose
                )
                result = exp.solve()
                if result == 'infeasible':
                    continue
                else:
                    self.results += result
        
            idx = np.argsort(-np.array(self.results))[:1000]
        else:
            idx = np.random.choice(len(self.df_lineups), size=max_, replace=False)

        exp = ExposureEnforcer(
            lineup_mtx.iloc[idx, :], 
            n_lineups_to_optimize, self.df.loc[players], var_multiple, verbose=verbose)
        self.result = exp.solve()
        self.df_optimal = self.df_lineups.loc[np.where(np.array(self.result).astype(int))[0]]
        self.df_optimal.reset_index(0, inplace=True, drop=True)
        return self.df_optimal

    def enforce_exposures_auto(self, var_multiple, cov_penalty, n_lineups_to_optimize, verbose=False):
        max_ = np.minimum(1000, len(self.df_lineups))
        iterations = int(np.ceil(len(self.df_lineups) / max_))
        self.results = []
        if iterations > 1:
            frac_to_select = 1000 / ((iterations - 1) * 1000 + len(self.df_lineups) % 1000)
            frac_to_select = min(0.3, frac_to_select)
            for k in range(iterations):
                print('Iteration {} of {}'.format(k+1, iterations+1))
                iter_df_lineups = self.df_lineups.iloc[max_ * k: max_ * (k+1)]
                n_to_select = frac_to_select * len(iter_df_lineups)
                exp = ExposureEnforcerAuto(
                    iter_df_lineups['mean'],
                    self.lineup_cov.iloc[
                        max_ * k: max_ * (k+1), max_ * k: max_ * (k+1)
                    ], 
                    round(n_to_select), var_multiple, 
                    cov_penalty, continuous='all', verbose=verbose)
                result = exp.solve()
                if result == 'infeasible':
                    continue
                else:
                    self.results += result
        
            idx = np.argsort(-np.array(self.results))[:1000]
        else:
            idx = np.arange(max_)

        if iterations > 1:
            print('Iteration {} of {}'.format(iterations+1, iterations+1))
        exp = ExposureEnforcerAuto(
            self.df_lineups.iloc[idx]['mean'],
            self.lineup_cov.iloc[idx, idx], n_lineups_to_optimize, 
            var_multiple, cov_penalty, continuous='all', verbose=verbose)
        self.result = exp.solve()
        idx = np.argsort(-np.array(self.result))[:n_lineups_to_optimize]
        self.df_optimal = self.df_lineups.loc[idx]
        self.df_optimal.reset_index(0, inplace=True, drop=True)
        return self.df_optimal

    def _get_lineup_salaries(self):
        self.df.index = self.df.Name
        df_sal = pd.DataFrame(
            columns=self.POSITION_COLS,
            index=self.df_lineups.index
        )
        for col in self.POSITION_COLS:
            df_sal[col] = self.df_lineups[col].map(
                self.df['Salary'].to_dict()
            )
        self.df_lineups['salary'] = df_sal.sum(axis=1)

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

    def __init__(self, lineup_mtx, n_lineups_to_optimize, df, var_multiple, time_limit=60, verbose=False):
        self.lineup_mtx = lineup_mtx
        self.n_lineups_to_optimize = n_lineups_to_optimize
        self.df = df
        self.var_multiple = var_multiple
        self.time_limit = time_limit
        self.verbose = verbose

        self._provision_constraints_from_df()
        self._construct_lp()

    def _provision_constraints_from_df(self):
        min_exp = self.df['min_exp'].values
        self.min_exp = 1.0 * np.floor(min_exp * self.n_lineups_to_optimize)
        max_exp = self.df['max_exp'].values
        self.max_exp = 1.0 * np.ceil(max_exp * self.n_lineups_to_optimize)
        self.mean = self.lineup_mtx['mean'].values
        self.std = self.lineup_mtx['std'].values
        self.n_lineups = len(self.lineup_mtx)
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
                    val=self.lineup_mtx.loc[:, self.players[j]].values * 1.0)],
                rhs=[self.min_exp[j]],
                names=["exp_limit_low_"+str(j)],
                senses=["G"])
            self.lp.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=index, 
                    val=self.lineup_mtx.loc[:, self.players[j]].values * 1.0)],
                rhs=[self.max_exp[j]],
                names=["exp_limit_high_"+str(j)],
                senses=["L"])
        self.lp.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=index, 
                val=[1.0]*self.n_lineups)],
            rhs=[self.n_lineups_to_optimize],
            names=["lineup_limit"],
            senses=["L"])

    def solve(self):
        self.lp.solve()
        if self.lp.solution.is_primal_feasible():
            self.result = self.lp.solution.get_values()
        else:
            print('Exposure enforcement is infeasible with the current lineups.')
            self.result = "infeasible"

        return self.result

class ExposureEnforcerAuto:
    
    def __init__(
        self, lineup_mean, lineup_cov, n_lineups_to_optimize, 
        var_multiple, cov_penalty, continuous=None, time_limit=60, verbose=False):

        self.lineup_mean = lineup_mean
        self.lineup_cov = lineup_cov
        self.n_lineups_to_optimize = n_lineups_to_optimize
        self.var_multiple = var_multiple
        self.cov_penalty = cov_penalty
        self.continuous = continuous
        if continuous is None:
            self.continuous = []
        elif continuous == 'all':
            self.continuous = list(range(len(lineup_mean)))

        self.time_limit = time_limit
        self.verbose = verbose

        self._provision_constraints_from_df()
        self._construct_lp()

    def _provision_constraints_from_df(self):
        self.mean = self.lineup_mean.values
        self.std = np.sqrt(np.diag(self.lineup_cov.values))
        self.cov = self.lineup_cov.values
        self.n_lineups = len(self.lineup_cov)

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
                names=["x_" + str(i)], ub=[1.0])
            self.lp.objective.set_linear(
                "x_" + str(i), 
                (self.mean[i] + \
                    (self.cov_penalty + self.var_multiple) * self.std[i]) / \
                        self.n_lineups_to_optimize)
            if i in self.continuous:
                var_type = self.lp.variables.type.continuous
            else:
                var_type = self.lp.variables.type.binary

            self.lp.variables.set_types(
                "x_" + str(i), var_type)
            
        
        quadratics = []
        for i in range(self.n_lineups):
            for j in range(self.n_lineups):
                if j > i:
                    continue

                quadratics += [(
                    "x_" + str(i), "x_" + str(j),
                    - self.cov_penalty * self.cov[i, j] / self.n_lineups_to_optimize
                )]
        self.lp.objective.set_quadratic_coefficients(quadratics)
        self.lp.objective.set_sense(self.lp.objective.sense.maximize)

    def _get_constraints(self):
        index = ["x_" + str(i) for i in range(self.n_lineups)]
        self.lp.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=index, 
                val=[1.0] * self.n_lineups)],
            rhs=[self.n_lineups_to_optimize],
            names=["lineup_limit"],
            senses=["L"])

    def solve(self):
        self.lp.solve()
        if self.lp.solution.is_primal_feasible():
            self.result = self.lp.solution.get_values()
        else:
            self.result = "infeasible"

        return self.result