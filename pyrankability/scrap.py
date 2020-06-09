def C_count(team1_names,team1_scores,team2_names,team2_scores,violation_map,games=None):
    assert np.all(np.array([len(team1_scores),len(team2_scores),len(team2_names),len(team1_names)]) == len(team1_scores))
    game_df = pd.concat([team1_names,team1_scores,team2_names,team2_scores],axis=1)
    if games is not None:
        game_df = game_df.loc[games]
    game_df = game_df.set_index([team1_names.name,team2_names.name])
    display(game_df)
    all_names = list(np.unique(list(team1_names)+list(team2_names)))
    sorted(all_names)
    C = pd.DataFrame(np.zeros((len(all_names),len(all_names))),columns=all_names,index=all_names)
    reset_game_df = game_df.reset_index()
    for index,row in game_df.iterrows():
        teami,teamj = index
        d_ij = game_df.loc[index,team1_scores.name] - game_df.loc[index,team2_scores.name]
        mask = reset_game_df[team1_names.name] == teamj
        mask.index = game_df.index
        if mask.any(): # did team k play team j
            games_ij = game_df.loc[mask]
            for index2, row2 in games_ij.iterrows():
                teamj,teamk = index2
                if teamk == teami:
                    continue
                d_kj = game_df.loc[index2,team2_scores.name] - game_df.loc[index2,team1_scores.name]
                for v1 in d_ij:
                    for v2 in d_kj: # all pairs check
                        C.loc[teami,teamk] += violation_map(v1,v2)
        # Annoying but for now I'm copying the above
        mask = reset_game_df[team2_names.name] == teamj
        mask.index = game_df.index
        if mask.any(): # did team k play team j
            games_ij = game_df.loc[mask]
            for index2, row2 in games_ij.iterrows():
                teamk,teamj = index2
                if teamk == teami:
                    continue
                d_kj = game_df.loc[index2,team1_scores.name] - game_df.loc[index2,team2_scores.name]
                for v1 in d_ij:
                    for v2 in d_kj: # all pairs check
                        C.loc[teami,teamk] += violation_map(v1,v2)
    return C

def C_difference(D,threshold=0,decimals=1):
    names = D.columns
    D = D.values
    Dcopy = np.copy(D).astype(float)
    Dcopy[D==0]=np.nan
    Dmeans_col = np.nanmean(Dcopy,axis=0)
    Dmeans_row = np.nanmean(Dcopy,axis=1)
    Dstd_col = np.nanstd(Dcopy,axis=0)
    Dstd_row = np.nanstd(Dcopy,axis=1)
    c = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            mask1 = D[:,j]-D[:,i]>threshold
            mask2 = D[j,:]-D[i,:]>threshold
            Dai = np.nan_to_num(np.round(10**decimals*(Dcopy[:,i]-Dmeans_col[i])/Dstd_col[i])/10**decimals,0)
            Daj = np.nan_to_num(np.round(10**decimals*(Dcopy[:,j]-Dmeans_col[j])/Dstd_col[j])/10**decimals,0)
            Dia = np.nan_to_num(np.round(10**decimals*(Dcopy[i,:]-Dmeans_row[i])/Dstd_row[i])/10**decimals,0)
            Dja = np.nan_to_num(np.round(10**decimals*(Dcopy[j,:]-Dmeans_row[j])/Dstd_row[j])/10**decimals,0)
            c[i,j] = np.sum(mask1*(Daj-Dai)) + np.sum(mask2*(Dja-Dia))
    return pd.DataFrame(c,index=names,columns=names)

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
    counts = pd.DataFrame(np.zeros((len(all_names),len(all_names))),columns=all_names,index=all_names)    
    for game in games:
        team1 = team1_names.loc[game]
        team2 = team2_names.loc[game]
        score1 = score_func(game,team1,team1_scores.loc[game])
        score2 = score_func(game,team2,team2_scores.loc[game])
        if abs(score1 - score2) >= threshold:
            D.loc[team1,team2] += score1
            D.loc[team2,team1] += score2
            counts.loc[team1,team2] += 1
            counts.loc[team2,team1] += 1
    inxs = np.where(counts > 0)
    D = D/counts
    D = D.fillna(0)
    if as_int:
        D = D.astype(int)
    return D

def D_point_ratio(team1_names,team1_scores,team2_names,team2_scores,
                  games=None,as_int=False,threshold=0,
                  score_func=lambda game,team,score: score, # default is just to return the score
                  decimals=1):
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
                D.iloc[i,j],D.iloc[j,i] = round(10**decimals*D.iloc[i,j]/D.iloc[j,i])/10**decimals,round(10**decimals*D.iloc[j,i]/D.iloc[i,j])/10**decimals
    if as_int:
        D = D.astype(int)
    return D

def solve(S_orig,indices = None, method=["lop","hillside"][1],num_random_restarts=0,lazy=False,verbose=False,find_pair=False,cont=False):
    n = c_orig.shape[0]
    
    temp_dir = tempfile.mkdtemp(dir="/dev/shm") # try to write this model to memory
    
    if indices is None:
        indices = list(range(n))
    sorted(indices)
    
    try:
    
        Pfirst = []
        Pfinal = []
        objs = []
        xs = []
        pair_Pfirst = []
        pair_Pfinal = []
        pair_objs = []
        pair_xs = []
        first_k = None
        for ix in range(num_random_restarts+1):
            if ix > 0:
                perm_inxs = np.random.permutation(range(c_orig.shape[0]))
                c = c_orig[perm_inxs,:][:,perm_inxs]
            else:
                perm_inxs = np.arange(n)
                c = copy.deepcopy(c_orig)

            model_file = os.path.join(temp_dir,"model.mps")
            if os.path.isfile(model_file):
                AP = read(model_file)
                x = {}
                for i in range(n-1):
                    for j in range(i+1,n):
                        x[i,j] = AP.getVarByName("x(%s,%s)"%(i,j))
            else:
                AP = Model(method)

                x = {}

                nvars = 0
                indices_hash = {}
                for iix,i in enumerate(indices):
                    indices_hash[i] = True
                for i in range(n-1):
                    for j in range(i+1,n):
                        if i in indices_hash and j in indices_hash:
                            if cont == True:
                                x[i,j] = AP.addVar(lb=0,vtype="C",ub=1,name="x(%s,%s)"%(i,j)) #continuous
                            else:
                                x[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="x(%s,%s)"%(i,j)) #binary
                        elif i in indices_hash:
                            x[i,j] = 1
                        else:
                            x[i,j] = 0
                        nvars += 1

                AP.update()
                ncons = 0
                for i in range(n):
                    for j in range(i+1,n):
                        for k in range(j+1,n):
                            trans_cons = []
                            trans_cons.append(AP.addConstr(x[i,j] + x[j,k] - x[i,k] <= 1))
                            trans_cons.append(AP.addConstr(x[i,j] + x[j,k] - x[i,k] >= 0))
                            ncons += 2
                            if lazy:
                                for cons in trans_cons:
                                    cons.setAttr(GRB.Attr.Lazy,1)
                                    
                print(len(x),nvars,ncons)

                AP.update()
                AP.write(model_file)
            if first_k is not None:
                if method == 'lop':
                    AP.addConstr(quicksum((D[i,j]-D[j,i])*x[i,j]+D[j,i] for i in range(n-1) for j in range(i+1,n)) == first_k)
                elif method == 'hillside':
                    AP.setObjective(quicksum((c[i,j]-c[j,i])*x[i,j]+c[j,i] for i in range(n-1) for j in range(i+1,n)),GRB.MINIMIZE)

            tic = time.perf_counter()
            if method == 'lop':
                AP.setObjective(quicksum((D[i,j]-D[j,i])*x[i,j]+D[j,i] for i in range(n-1) for j in range(i+1,n)),GRB.MAXIMIZE)
            elif method == 'hillside':
                AP.setObjective(quicksum((c[i,j]-c[j,i])*x[i,j]+c[j,i] for i in indices for j in range(i+1,n)),GRB.MINIMIZE)
            AP.setParam( 'OutputFlag', verbose )
            AP.update()
            toc = time.perf_counter()
            if verbose:
                print(f"Updating opjective in {toc - tic:0.4f} seconds")

            if verbose:
                print('Start optimization %d'%ix)
            tic = time.perf_counter()
            AP.params.Threads=7
            AP.update()
            if cont:
                AP.Params.Method = 2
                AP.Params.Crossover = 0    
                AP.update()
            AP.optimize()
            toc = time.perf_counter()
            if verbose:
                print(f"Optimization in {toc - tic:0.4f} seconds")
                print('End optimization %d'%ix)

            k=AP.objVal
            if first_k is None:
                first_k = k

            P = []
            sol_x = get_sol_x_by_x(x,n,cont=cont)()
            orig_sol_x = sol_x
            reorder = np.argsort(perm_inxs)
            sol_x = sol_x[np.ix_(reorder,reorder)]
            r = np.sum(sol_x,axis=0)
            ranking = np.argsort(r)
            P.append(tuple(ranking))
            xs.append(sol_x)

            if ix == 0:
                Pfirst = P
                xfirst = get_sol_x_by_x(x,n,cont=cont)()

            Pfinal.extend(P)
            if method == 'lop':
                objs.append(np.sum(D_orig*sol_x))
            elif method == 'hillside':
                objs.append(np.sum(c_orig*sol_x))

            if find_pair:
                AP = read(model_file)
                x = {}
                for i in range(n-1):
                    for j in range(i+1,n):
                        x[i,j] = AP.getVarByName("x(%s,%s)"%(i,j))
                if method == 'lop':
                    AP.addConstr(quicksum((D[i,j]-D[j,i])*x[i,j]+D[j,i] for i in range(n-1) for j in range(i+1,n))==first_k)
                elif method == 'hillside':
                    AP.addConstr(quicksum((c[i,j]-c[j,i])*x[i,j]+c[j,i] for i in range(n-1) for j in range(i+1,n))==first_k)                
                AP.update()
                u={}
                v={}
                b={}
                for i in range(n-1):
                    for j in range(i+1,n):
                        u[i,j] = AP.addVar(name="u(%s,%s)"%(i,j),lb=0)
                        v[i,j] = AP.addVar(name="v(%s,%s)"%(i,j),lb=0)
                        b[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="b(%s,%s)"%(i,j))
                AP.update()
                for i in range(n-1):
                    for j in range(i+1,n):
                        AP.addConstr(u[i,j] - v[i,j] == x[i,j] - orig_sol_x[i,j])
                        AP.addConstr(u[i,j] <= b[i,j])
                        AP.addConstr(v[i,j] <= 1 - b[i,j])
                AP.update()

                AP.setObjective(quicksum(u[i,j]+v[i,j] for i in range(n-1) for j in range(i+1,n)),GRB.MAXIMIZE)
                AP.setParam( 'OutputFlag', verbose )
                AP.update()

                if verbose:
                    print('Start pair optimization %d'%ix)
                tic = time.perf_counter()

                if cont:
                    AP.Params.Method = 2
                    AP.Params.Crossover = 0    
                    AP.update()

                AP.optimize()
                toc = time.perf_counter()
                if verbose:
                    print(f"Optimization in {toc - tic:0.4f} seconds")
                    print('End optimization %d'%ix)

                P = []
                sol_x = get_sol_x_by_x(x,n,cont=cont)()[np.ix_(reorder,reorder)]
                sol_u = get_sol_x_by_x(u,n)()[np.ix_(reorder,reorder)]
                sol_v = get_sol_x_by_x(v,n)()[np.ix_(reorder,reorder)]
                r = np.sum(sol_x,axis=0)
                ranking = np.argsort(r)
                P.append(tuple(ranking))
                pair_xs.append(sol_x)
                if method == 'lop':
                    k = np.sum(np.sum(D_orig*sol_x))
                elif method == 'hillside':
                    k = np.sum(np.sum(c_orig*sol_x))

                if ix == 0:
                    pair_Pfirst = P
                    pair_xfirst = get_sol_x_by_x(x,n)() 

                pair_Pfinal.extend(P)
                pair_objs.append(k)

        details = {"Pfirst": Pfirst, "P":Pfinal,"x": xfirst,"objs":objs,"xs":xs}
        pair_details = None
        if find_pair:
            pair_details = {"Pfirst": pair_Pfirst, "P":pair_Pfinal,"x": pair_xfirst,"objs":pair_objs,"xs":pair_xs}
        details["pair_details"] = pair_details

        details['method'] = method
        details['indices'] = indices
    finally:
        shutil.rmtree(temp_dir)
        
    return k,details