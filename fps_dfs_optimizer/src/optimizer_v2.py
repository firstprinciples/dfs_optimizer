import numpy as np
import pandas as pd
from scipy.stats import norm
import cplex
from cplex.callbacks import SolveCallback

def build_lineups(N_lineups,
                  order,
                  projections, 
                  variances, 
                  salaries, 
                  risk_penalty, 
                  positions, 
                  exclusive_mat,
                  guards,
                  forwards,
                  max_exp,
                  min_exp,
                  cap=50, 
                  exclusive_lim=[3.0, 3.0, 3.0, 3.0, 2.0],
                  max_players=8.0, 
                  time_limit=60.0,
                  suppress=True
                 ):
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

def lineup_sort(tiptime,position_mtx,max_players=8,time_limit=60):
    ones = np.ones((max_players)).astype(float)
    ones = ones.tolist()
    lineup_sort_program = cplex.Cplex()
    for i in range(max_players):
        for j in range(max_players):
            lineup_sort_program.variables.add(names= ["x"+str(i)+str(j)])
            lineup_sort_program.variables.set_types(i*max_players+j, lineup_sort_program.variables.type.binary)
    for i in range(max_players):
        for j in range(5):
            lineup_sort_program.objective.set_linear([(i*max_players+j, 0)])
        for j in range(5,max_players-1):
            lineup_sort_program.objective.set_linear([(i*max_players+j, tiptime[i])])
        lineup_sort_program.objective.set_linear([(i*max_players+7, 1.1*tiptime[i])])
    lineup_sort_program.objective.set_sense(lineup_sort_program.objective.sense.maximize)
    for i in range(max_players):
        lineup_sort_program.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [i*max_players+j for j in range(max_players)], val= ones)],
            rhs= [1.0],
            names = ["player_limit"+str(i)],
            senses = ["E"]
        )
    for j in range(max_players):
        lineup_sort_program.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [i*max_players+j for i in range(max_players)], val= ones)],
            rhs= [1.0],
            names = ["position_limit"+str(j)],
            senses = ["E"]
        )
    for i in range(max_players):
        for j in range(max_players):
            lineup_sort_program.linear_constraints.add(
                lin_expr= [cplex.SparsePair(ind= [i*max_players+j], val= [1.0])],
                rhs= [position_mtx[i,j]],
                names = ["position_constraint"+str(i)+str(j)],
                senses = ["L"]
            )
    lineup_sort_program.parameters.timelimit = time_limit
    lineup_sort_program.set_log_stream(None)
    lineup_sort_program.set_error_stream(None)
    lineup_sort_program.set_warning_stream(None)
    lineup_sort_program.set_results_stream(None)
    lineup_sort_program.solve()
    
    if lineup_sort_program.solution.is_primal_feasible():
        output = np.reshape(lineup_sort_program.solution.get_values(),(max_players,max_players))
    else:
        output = "infeasible"
                                                  
    return output

def lineup_sort_locked(tiptime,position_mtx,locked_mtx,max_players=8,time_limit=60):
    ones = np.ones((max_players)).astype(float)
    ones = ones.tolist()
    lineup_sort_program = cplex.Cplex()
    for i in range(max_players):
        for j in range(max_players):
            lineup_sort_program.variables.add(names= ["x"+str(i)+str(j)])
            lineup_sort_program.variables.set_types(i*max_players+j, lineup_sort_program.variables.type.binary)
    for i in range(max_players):
        for j in range(5):
            lineup_sort_program.objective.set_linear([(i*max_players+j, 0)])
        for j in range(5,max_players-1):
            lineup_sort_program.objective.set_linear([(i*max_players+j, tiptime[i])])
        lineup_sort_program.objective.set_linear([(i*max_players+7, 1.1*tiptime[i])])
    lineup_sort_program.objective.set_sense(lineup_sort_program.objective.sense.maximize)
    for i in range(max_players):
        lineup_sort_program.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [i*max_players+j for j in range(max_players)], val= ones)],
            rhs= [1.0],
            names = ["player_limit"+str(i)],
            senses = ["E"]
        )
    for j in range(max_players):
        lineup_sort_program.linear_constraints.add(
            lin_expr= [cplex.SparsePair(ind= [i*max_players+j for i in range(max_players)], val= ones)],
            rhs= [1.0],
            names = ["position_limit"+str(j)],
            senses = ["E"]
        )
    for i in range(max_players):
        for j in range(max_players):
            lineup_sort_program.linear_constraints.add(
                lin_expr= [cplex.SparsePair(ind= [i*max_players+j], val= [1.0])],
                rhs= [position_mtx[i,j]],
                names = ["position_constraint"+str(i)+str(j)],
                senses = ["L"]
            )
    for i in range(max_players):
        for j in range(max_players):
            lineup_sort_program.linear_constraints.add(
                lin_expr= [cplex.SparsePair(ind= [i*max_players+j], val= [1.0])],
                rhs= [locked_mtx[i,j]],
                names = ["locked_constraint"+str(i)+str(j)],
                senses = ["G"]
            )
    lineup_sort_program.parameters.timelimit = time_limit
    lineup_sort_program.set_log_stream(None)
    lineup_sort_program.set_error_stream(None)
    lineup_sort_program.set_warning_stream(None)
    lineup_sort_program.set_results_stream(None)
    lineup_sort_program.solve()
    
    if lineup_sort_program.solution.is_primal_feasible():
        output = np.reshape(lineup_sort_program.solution.get_values(),(max_players,max_players))
    else:
        output = "infeasible"
                                                  
    return output

def sort_lineups(df, df_lineups, lineup_cols):
    pos_cols = ['pg', 'sg', 'sf', 'pf', 'c', 'guards', 'forwards', 'util']
    final_cols = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    df_player_lineups = pd.DataFrame(columns=pos_cols)
    df_ID_lineups = pd.DataFrame(columns=final_cols)
    for lineup in lineup_cols:
        idx = df_lineups.index[np.where(df_lineups[lineup]==1)[0]]
        positions_mtx = np.concatenate((df.loc[idx,pos_cols[:-1]].values,np.ones((len(idx),1))),axis=1)
        current_lineup = lineup_sort(df.loc[idx,'tip time'].values,
                                     positions_mtx)
        if current_lineup == 'infeasible':
            print(lineup+' '+current_lineup)
            df_current_lineup = pd.DataFrame(columns=pos_cols)
            df_current_IDs = pd.DataFrame(columns=final_cols)
        else:
            player_lineup = []
            ID_lineup = []
            for pos in range(len(pos_cols)):
                index = list(idx)[np.where(current_lineup[:,pos]==1)[0][0]]
                player_lineup += [df.loc[index,'Player']]
                ID_lineup += [df.loc[index,'DK ID']]
            df_current_lineup = pd.DataFrame(np.expand_dims(np.array(player_lineup),axis=0),columns=pos_cols)
            df_current_IDs = pd.DataFrame(np.expand_dims(np.array(ID_lineup),axis=0),columns=final_cols)
        df_player_lineups = pd.concat((df_player_lineups, df_current_lineup),axis=0)
        df_ID_lineups = pd.concat((df_ID_lineups, df_current_IDs),axis=0)
        df_player_lineups.index = lineup_cols[:df_player_lineups.shape[0]]
        df_ID_lineups.index = lineup_cols[:df_player_lineups.shape[0]]
    return df_player_lineups, df_ID_lineups

def sort_lineups_locks(df, df_lineups, locked, lineup_cols):
    pos_cols = ['pg', 'sg', 'sf', 'pf', 'c', 'guards', 'forwards', 'util']
    final_cols = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    df_player_lineups = pd.DataFrame(columns=pos_cols)
    df_ID_lineups = pd.DataFrame(columns=final_cols)
    for lineup in lineup_cols:
        idx = df_lineups.index[np.where(df_lineups[lineup]==1)[0]]
        positions_mtx = np.concatenate((df.loc[idx,pos_cols[:-1]].values,np.ones((len(idx),1))),axis=1)
        locked_mtx = np.zeros((len(pos_cols),len(pos_cols)))
        for j, player in enumerate(locked):
            if player != '':
                locked_mtx[np.where(idx == player)[0],j] = 1
        current_lineup = lineup_sort_locked(df.loc[idx,'tip time'].values,
                                            positions_mtx, locked_mtx)
        if current_lineup == 'infeasible':
            print(lineup+' '+current_lineup)
            df_current_lineup = pd.DataFrame(columns=pos_cols)
            df_current_IDs = pd.DataFrame(columns=final_cols)
        else:
            player_lineup = []
            ID_lineup = []
            for pos in range(len(pos_cols)):
                index = list(idx)[np.where(current_lineup[:,pos]==1)[0][0]]
                player_lineup += [df.loc[index,'Player']]
                ID_lineup += [df.loc[index,'DK ID']]
            df_current_lineup = pd.DataFrame(np.expand_dims(np.array(player_lineup),axis=0),columns=pos_cols)
            df_current_IDs = pd.DataFrame(np.expand_dims(np.array(ID_lineup),axis=0),columns=final_cols)
        df_player_lineups = pd.concat((df_player_lineups, df_current_lineup),axis=0)
        df_ID_lineups = pd.concat((df_ID_lineups, df_current_IDs),axis=0)
        df_player_lineups.index = lineup_cols[:df_player_lineups.shape[0]]
        df_ID_lineups.index = lineup_cols[:df_player_lineups.shape[0]]
    return df_player_lineups, df_ID_lineups

def lineup_monte_carlo(df_lineups, df, cols, sims=10000, players=8):
    
    df_lineups['Mean'] = [0]*df_lineups.shape[0]
    df_lineups['std'] = [0]*df_lineups.shape[0]
    df_lineups['Actual'] = [0]*df_lineups.shape[0]
    
    for lineup in list(df_lineups.index):
        dists = norm().rvs(size=(sims,players))
        idx = np.array(df.index)[np.where(df['Player'].isin(df_lineups.loc[lineup,cols]))[0]]
        while idx.shape[0] < 8:
            idx = np.append(idx, list(df.index)[-1]) ##Assumes that df is sorted by ppd. Thus, df.loc[idx,'DK_points'] is basically 0
        scores = dists @ np.expand_dims(df.loc[idx,'DK std'].values,axis=1) + np.expand_dims(df.loc[idx,'DK_points'].sum(),axis=1)
        actual_score = np.expand_dims(df.loc[idx,'DK actual'].sum(),axis=1)
        df_lineups.loc[lineup,'Mean'] = np.mean(scores)
        df_lineups.loc[lineup,'std'] = np.std(scores)
        df_lineups.loc[lineup,'Actual'] = actual_score
    
    return df_lineups

def get_lineups_from_dkentries(filename,df,total_lineups):
    imported_lineups = pd.read_csv("../data/external/"+filename, sep=',').iloc[:total_lineups,:12]
    N_imported_lineups = np.where(imported_lineups['PG']==0)[0][0]
    lineup_name = np.zeros((N_imported_lineups,8)).astype(str)
    for i in range(N_imported_lineups):
        for n, pos in enumerate(df_ID_lineups.columns):
            lineup_name[i,n] = df.loc[np.where(imported_lineups.loc[i,pos]==df['DK ID'])[0][0],'Player']
    return lineup_name

def rebuild_dkentries(filename,df_ID_lineups,total_lineups):
    final_cols = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    imported_lineups = pd.read_csv("../data/external/"+filename, sep=',').iloc[:total_lineups,:12]
    N_imported_lineups = np.where(imported_lineups['PG']==0)[0][0]
    imported_lineups_new = np.concatenate((imported_lineups.values[:N_imported_lineups,4:],df_ID_lineups.values),axis=0)
    imported_lineups.iloc[:imported_lineups_new.shape[0],4:] = imported_lineups_new[:total_lineups]
    imported_lineups[final_cols] = imported_lineups[final_cols].astype(int)
    imported_lineups['Contest ID'] = imported_lineups['Contest ID'].astype(int)
    imported_lineups['Entry ID'] = imported_lineups['Entry ID'].astype(int)
    return imported_lineups

def get_player_count(df_player_lineups):
    player_list = df_player_lineups.values.ravel()
    players_in_lineups = np.unique(player_list)
    player_count = []
    for player in players_in_lineups:
        player_count += [np.where(player_list == player)[0].shape[0]]
    return pd.DataFrame(player_count,columns=['Number of entries'],
                        index=np.unique(player_list)).sort_values(by='Number of entries',ascending=False)