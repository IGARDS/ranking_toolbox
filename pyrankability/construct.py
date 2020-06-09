import multiprocessing

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

def link(game_df,team_i,team_k,value_map):
    team_j_i = "team1"
    if team_i == "team1":
        team_j_i = "team2"
    team_j_k = "team2"
    if team_k == "team2":
        team_j_k = "team1"
    left = game_df.copy()
    new_cols=[]
    for c in left.columns:
        new_cols.append(c.replace(team_j_i,"team_j_i").replace(team_i,"team_i").replace("game","game_i_j"))
    left.columns=new_cols
    left=left.set_index("team_j_i_name")
    
    right = game_df.copy()
    new_cols=[]
    for c in right.columns:
        new_cols.append(c.replace(team_j_k,"team_j_k").replace(team_k,"team_k").replace("game","game_k_j"))
    right.columns=new_cols
    right = right.set_index("team_j_k_name")
    linked = left.join(right,how='inner') 
    linked.index.name="team_j_name"
    linked=linked.reset_index()
    #linked = linked.loc[linked.team_i_name != linked.team_k_name]
    return linked

def V_count_vectorized(game_df,value_map):
    all_names = list(np.unique(list(game_df["team1_name"])+list(game_df["team2_name"])))
    game_df["game"] = list(game_df.index)
    game_df = game_df.set_index(["team1_name",'team2_name'])
    sorted(all_names)
    reset_game_df = game_df.reset_index()

    linked = link(reset_game_df,"team1","team1",value_map)
    linked=linked.append(link(reset_game_df,"team2","team2",value_map))
    linked=linked.append(link(reset_game_df,"team2","team1",value_map))
    linked.index=list(range(len(linked)))
    linked["games"] = linked["game_i_j"].astype(str)+","+linked["game_k_j"].astype(str)
    linked.loc[linked["game_k_j"] < linked["game_i_j"],"games"] = linked["game_k_j"].astype(str)+","+linked["game_i_j"].astype(str)
    linked = linked.drop_duplicates(subset='games', keep='first')
    return value_map(linked).unstack()

# S is basically our traditional D, but I'm now calling it support to show how we are measuring the support of team i above team j
# If you constructed a violation matrix instead of a support matrix, then just pass -V
# Threshold is the level of difference in support that is considered a violation of hillside
# Important note, please make S.iloc[i,j] = np.NaN if you don't have any information
def C_count(S,threshold=0):
    names = S.columns
    S = S.values
    c = np.zeros(S.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            if i == j:
                continue
            mask1 = np.abs(S[:,j]-S[:,i])>threshold # smooth things out and don't worry about warnings here
            mask2 = np.abs(S[j,:]-S[i,:])>threshold # smooth things out and don't worry about warnings here
            
            c[i,j] = np.sum(S[mask1,j]-S[mask1,i]<0) + np.sum(S[i,mask2]-S[j,mask2]<0)
    return pd.DataFrame(c,index=names,columns=names)

# S is basically our traditional D, but I'm now calling it support to show how we are measuring the support of team i above team j
# If you constructed a violation matrix instead of a support matrix, then just pass -V
# Threshold is the level of difference in support that is considered a violation of hillside
# Important note, please make S.iloc[i,j] = np.NaN if you don't have any information
def C_count(S,threshold=0):
    names = S.columns
    S = S.values
    c = np.zeros(S.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            if i == j:
                continue
            mask1 = np.abs(S[:,j]-S[:,i])>threshold # smooth things out and don't worry about warnings here
            mask2 = np.abs(S[j,:]-S[i,:])>threshold # smooth things out and don't worry about warnings here
            
            c[i,j] = np.sum(S[mask1,j]-S[mask1,i]<0) + np.sum(S[i,mask2]-S[j,mask2]<0)
    return pd.DataFrame(c,index=names,columns=names)