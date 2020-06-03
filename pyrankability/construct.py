import multiprocessing

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

# Calculates how many times you are above threshold (rounded)
def D_point_differential(team1_names,team1_scores,team2_names,team2_scores,
                                   games=None,as_int=False,threshold=0,
                                   score_func=lambda game,team,score: score, # default is just to return the score
                                   diff_to_dij_func=lambda score1,score2: score1-score2 # default is just to return the difference
                                  ):
    assert np.all(np.array([len(team1_scores),len(team2_scores),len(team2_names),len(team1_names)]) == len(team1_scores))
    if games is None:
        games = list(range(len(team1_scores)))
    all_names = list(np.unique(list(team1_names)+list(team2_names)))
    sorted(all_names)
    D = pd.DataFrame(np.zeros((len(all_names),len(all_names))),columns=all_names,index=all_names)
    for game in games:
        team1 = team1_names.loc[game]
        team2 = team2_names.loc[game]
        score1 = score_func(game,team1,team1_scores.loc[game])
        score2 = score_func(game,team2,team2_scores.loc[game])
        if (score1 - score2) >= threshold:
            D.loc[team1,team2] += diff_to_dij_func(score1,score2)
        elif (score2 - score1) >= threshold:
            D.loc[team2,team1] += diff_to_dij_func(score2,score1)
    if as_int:
        D = D.astype(int)
    return D

# Calculates how many times you are above threshold (rounded)
def D_point_score(team1_names,team1_scores,team2_names,team2_scores,
                                   games=None,as_int=False,threshold=0,
                                   score_func=lambda game,team,score: score, # default is just to return the score
                                  ):
    assert np.all(np.array([len(team1_scores),len(team2_scores),len(team2_names),len(team1_names)]) == len(team1_scores))
    if games is None:
        games = list(range(len(team1_scores)))
    all_names = list(np.unique(list(team1_names)+list(team2_names)))
    sorted(all_names)
    D = pd.DataFrame(np.zeros((len(all_names),len(all_names))),columns=all_names,index=all_names)
    for game in games:
        team1 = team1_names.loc[game]
        team2 = team2_names.loc[game]
        score1 = score_func(game,team1,team1_scores.loc[game])
        score2 = score_func(game,team2,team2_scores.loc[game])
        if abs(score1 - score2) >= threshold:
            D.loc[team1,team2] += score1
            D.loc[team2,team1] += score2
    if as_int:
        D = D.astype(int)
    return D

def D_point_ratio(team1_names,team1_scores,team2_names,team2_scores,
                                   games=None,as_int=False,threshold=0,
                                   score_func=lambda game,team,score: score, # default is just to return the score
                                  ):
    assert np.all(np.array([len(team1_scores),len(team2_scores),len(team2_names),len(team1_names)]) == len(team1_scores))
    if games is None:
        games = list(range(len(team1_scores)))
    all_names = list(np.unique(list(team1_names)+list(team2_names)))
    sorted(all_names)
    D = pd.DataFrame(np.zeros((len(all_names),len(all_names))),columns=all_names,index=all_names)
    for game in games:
        team1 = team1_names.loc[game]
        team2 = team2_names.loc[game]
        score1 = score_func(game,team1,team1_scores.loc[game])
        score2 = score_func(game,team2,team2_scores.loc[game])
        if abs(score1 - score2) >= threshold:
            D.loc[team1,team2] += score1
            D.loc[team2,team1] += score2
    for i in range(len(all_names)):
        for j in range(i+1,len(all_names)):
            if D.iloc[i,j] != 0 and D.iloc[j,i] != 0:
                D.iloc[i,j],D.iloc[j,i] = D.iloc[i,j]/D.iloc[j,i],D.iloc[j,i]/D.iloc[i,j]
    if as_int:
        D = D.astype(int)
    return D

def C_count(D,threshold=0):
    names = D.columns
    D = D.values
    c = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            mask = (np.abs(D[:,j]-D[:,i])>=threshold)
            c[i,j] = np.count_nonzero(mask*(D[:,j]-D[:,i]<0)) + np.count_nonzero(mask*(D[i,:]-D[j,:])<0) 
    return pd.DataFrame(c,index=names,columns=names)

def C_difference(D,threshold=0):
    names = D.columns
    D = D.values
    c = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            mask = (np.abs(D[:,j]-D[:,i])>=threshold)
            c[i,j] = np.sum(mask*(D[:,j]-D[:,i]<0)*(D[:,i]-D[:,j])) + np.sum(mask*(D[i,:]-D[j,:]<0)*(D[j,:]-D[i,:]))
    return pd.DataFrame(c,index=names,columns=names)
