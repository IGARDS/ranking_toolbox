def solve_outlier(D,orig_obj,orig_sol_x,method=["lop","hillside"][1],lazy=False,verbose=False) :
    n = D.shape[0]
    AP = Model(method)
    
    if method == 'hillside':
        c = C_count(D)

    x = {}

    for i in range(n-1):
        for j in range(i+1,n):
            x[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="x(%s,%s)"%(i,j)) #binary

    AP.update()
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                trans_cons = []
                trans_cons.append(AP.addConstr(x[i,j] + x[j,k] - x[i,k] <= 1))
                trans_cons.append(AP.addConstr(x[i,j] + x[j,k] - x[i,k] >= 0))
                if lazy:
                    for cons in trans_cons:
                        cons.setAttr(GRB.Attr.Lazy,1)

    AP.update()
    if method == 'lop':
        AP.addConstr(quicksum((D.iloc[i,j]-D.iloc[j,i])*x[i,j]+D.iloc[j,i] for i in range(n-1) for j in range(i+1,n))==orig_obj)
    elif method == 'hillside':
        AP.addConstr(quicksum((c.iloc[i,j]-c.iloc[j,i])*x[i,j]+c.iloc[j,i] for i in range(n-1) for j in range(i+1,n))==orig_obj)                
    AP.update()
    
    ij_1 = []
    ij_0 = []
    for i in range(n-1):
        for j in range(i+1,n):
            if orig_sol_x[i,j] == 1:
                ij_1.append((i,j))
            else:
                ij_0.append((i,j))
    AP.addConstr(quicksum(x[i,j]-orig_sol_x[i,j] for i,j in ij_0)+quicksum(orig_sol_x[i,j] - x[i,j] for i,j in ij_1) >= 1)                
    AP.update()

    AP.setParam( 'OutputFlag', verbose )
    AP.update()

    if verbose:
        print('Start pair optimization')
    tic = time.perf_counter()
    AP.optimize()
    toc = time.perf_counter()
    if verbose:
        print(f"Optimization in {toc - tic:0.4f} seconds")
        print('End optimization')

    sol_x = get_sol_x_by_x(x,n)()
    r = np.sum(sol_x,axis=0)
    ranking = np.argsort(r)
    perm = tuple([int(item) for item in ranking])
    
    details = {"obj":AP.objVal,"perm":perm,"x":sol_x}
    if method == 'hillside':
        details['c'] = c
        k = round(k)
    elif method == 'lop': # switch to delta
        Dre = D.values[perm,:][:,perm]
        #print(k,np.sum(np.triu(Dre)))
        k = np.sum(np.tril(Dre,k=-1))
        
    return k,details