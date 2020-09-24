import multiprocessing

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

def link(game_df,team_i,team_k):
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

    linked = link(reset_game_df,"team1","team1")
    linked=linked.append(link(reset_game_df,"team2","team2"))
    linked=linked.append(link(reset_game_df,"team2","team1"))
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

def support_map_vectorized_direct_indirect_weighted(linked,direct_thres=1,spread_thres=0,weight_indirect=0.5,verbose=False):
    # columns
    # 'team_j', 'team_i_name', 'team_i_score', 'team_i_H_A_N',
    # 'team_j_i_score', 'team_j_i_H_A_N', 'game_i_j', 'team_k_name',
    # 'team_k_score', 'team_k_H_A_N', 'team_j_k_score', 'team_j_k_H_A_N',
    # 'game_k_j'
    linked["direct"] = linked["game_i_j"] == linked["game_k_j"] #linked["team_i_name"] == linked["team_k_name"]
    linked["indirect"] = linked["team_i_name"] != linked["team_k_name"]
    # | (linked["team_i_name"] == linked["team_j_k_name"]) | (linked["team_k_name"] == linked["team_j_k_name"])
    for_index1 = linked[["team_i_name","team_k_name"]].copy()
    for_index1.loc[linked["direct"]] = linked.loc[linked["direct"],["team_i_name","team_j_name"]]
    for_index1.columns = ["team1","team2"]
    for_index2 = linked[["team_k_name","team_i_name"]].copy()
    for_index2.loc[linked["direct"]] = linked.loc[linked["direct"],["team_j_name","team_i_name"]]
    for_index2.columns = ["team1","team2"]
    index_ik = pd.MultiIndex.from_frame(for_index1,sortorder=0)
    index_ki = pd.MultiIndex.from_frame(for_index2,sortorder=0)
    
    #######################################
    # part to modify
    # direct
    d_ik = linked['team_i_score'] - linked['team_j_i_score']
    support_ik = (linked["direct"] & (d_ik > direct_thres)).astype(int)
    support_ki = (linked["direct"] & (d_ik < -direct_thres)).astype(int)

    # indirect
    d_ij = linked["team_i_score"] - linked["team_j_i_score"]
    d_kj = linked["team_k_score"] - linked["team_j_k_score"]
    
    # always a positive and it captures that if i beat j by 5 points and k beat j by 2 points then this spread is 3
    spread = np.abs(d_ij - d_kj) 
    
    support_ik += weight_indirect*((linked["indirect"]) & (d_ij > 0) & (d_kj < 0) & (spread > spread_thres)).astype(int)
    
    support_ki += weight_indirect*((linked["indirect"]) & (d_kj > 0) & (d_ij < 0) & (spread > spread_thres)).astype(int)
    
    # end part to modify
    #######################################    
    linked["support_ik"]=support_ik
    linked["index_ik"]=index_ik
    linked["support_ki"]=support_ki
    linked["index_ki"]=index_ki
        
    if verbose:
        print('Direct')
        print("Total:",sum(linked["direct"] & (linked["support_ik"]>0)) + sum(linked["direct"] & (linked["support_ki"]>0)),
              "ik:",sum(linked["direct"] & (linked["support_ik"]>0)), 
              "ki:",sum(linked["direct"] & (linked["support_ki"]>0)))
        print('Indirect')
        print("Total:",sum((~linked["direct"]) & (linked["support_ik"]>0)) + sum(~linked["direct"] & (linked["support_ki"]>0)),
              "ik:",sum((~linked["direct"]) & (linked["support_ik"]>0)), 
              "ki:",sum(~linked["direct"] & (linked["support_ki"]>0)))
    
    #indices_ik = linked.index[(linked["support_ik"] > linked["support_ki"]) & ~linked['direct']]    
    #indices_ki = linked.index[(linked["support_ik"] > linked["support_ki"]) & ~linked['direct']] 
    #linked.loc[~linked['direct']] = 0
    #linked.loc[indices_ik] = 0.5*(linked.loc[indices_ik]["support_ik"] - linked.loc[indices_ik]["support_ki"])
    #linked.loc[indices_ik] = 0.5*(linked.loc[indices_ik]["support_ik"] - linked.loc[indices_ik]["support_ki"])
    
    ret1 = linked.set_index(index_ik)["support_ik"]
    ret2 = linked.set_index(index_ki)["support_ki"]
    ret = ret1.append(ret2)
    ret = ret.groupby(level=[0,1]).sum()
    return ret