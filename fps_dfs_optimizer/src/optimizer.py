import numpy as np
import pandas as pd
from scipy.stats import norm
import cplex

class LineupOptimizer:

    def __init__(self, n_lineups):


def build_lineups(
    N_lineups, order, projections, 
    variances, salaries, risk_penalty, 
    positions, exclusive_mat, guards,
    forwards, max_exp, min_exp,
    cap=50, exclusive_lim=[3.0, 3.0, 3.0, 3.0, 2.0],
    max_players=8.0, time_limit=60.0,
    suppress=True):
    
    N_players = projections.shape[0]
    lineup_program = cplex.Cplex()
    lineup_program.parameters.timelimit.set(time_limit)
    for i in range(N_players):
        for j in range(N_lineups):
            lineup_program.variables.add(names= ["y"+str(i)+str(j)])
            lineup_program.variables.set_types(i*N_lineups+j, lineup_program.variables.type.binary)
            lineup_program.objective.set_linear([(i*N_lineups+j,
                                                  (1+order*(N_lineups-j-1))*(projections[i]-risk_penalty*variances[i]))])
    lineup_program.objective.set_sense(lineup_program.objective.sense.maximize)
    for j in range(N_lineups):
        lineup_program.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [i*N_lineups+j for i in range(N_players)], val= salaries)],
            rhs= [cap],
            names = ["cap_limit"],
            senses = ["L"]
        )
        lineup_program.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [i*N_lineups+j for i in range(N_players)], val= [1.0]*N_players)],
            rhs= [max_players],
            names = ["player_limit"],
            senses = ["E"]
        )
        for k in range(5):
            lineup_program.linear_constraints.add(
                lin_expr= [cplex.SparsePair(ind= [i*N_lineups+j for i in range(N_players)], val= positions[:,k])],
                rhs= [1.0],
                names = ["position_limit"],
                senses = ["G"]
            )
            lineup_program.linear_constraints.add(
                lin_expr= [cplex.SparsePair(ind= [i*N_lineups+j for i in range(N_players)], val= exclusive_mat[:,k])],
                rhs= [exclusive_lim[k]],
                names = ["exc_limit"],
                senses = ["L"]
            )
        lineup_program.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [i*N_lineups+j for i in range(N_players)], val= guards)],
            rhs= [3.0],
            names = ["guard_limit"],
            senses = ["G"]
        )
        lineup_program.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [i*N_lineups+j for i in range(N_players)], val= forwards)],
            rhs= [3.0],
            names = ["forward_limit"],
            senses = ["G"]
        )
    for i in range(N_players):
        lineup_program.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [i*N_lineups+j for j in range(N_lineups)], val= [1]*N_lineups)],
            rhs= [min_exp[i]],
            names = ["exp_limit_low"],
            senses = ["G"]
        )
        lineup_program.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [i*N_lineups+j for j in range(N_lineups)], val= [1]*N_lineups)],
            rhs= [max_exp[i]],
            names = ["exp_limit_high"],
            senses = ["L"]
        )
    if suppress:
        lineup_program.set_log_stream(None)
        lineup_program.set_error_stream(None)
        lineup_program.set_warning_stream(None)
        lineup_program.set_results_stream(None)
    lineup_program.solve()
    if lineup_program.solution.is_primal_feasible():
        output = np.reshape(lineup_program.solution.get_values(),(N_players,N_lineups))
    else:
        output = "infeasible"
    return output