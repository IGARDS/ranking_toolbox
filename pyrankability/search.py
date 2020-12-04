# -*- coding: utf-8 -*-
import itertools
import copy
import multiprocessing
import tempfile
import os
import shutil
import time

import numpy as np
from gurobipy import *
from joblib import Parallel, delayed

from .rank import *

from .construct import *

# TODO: change all solve returns to use obj instead of k

def solve_max_tau(D,orig_k,orig_sol_x,method=["lop","hillside"][1],lazy=False,verbose=False) :
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
        AP.addConstr(quicksum((D.iloc[i,j]-D.iloc[j,i])*x[i,j]+D.iloc[j,i] for i in range(n-1) for j in range(i+1,n))==orig_k)
    elif method == 'hillside':
        AP.addConstr(quicksum((c.iloc[i,j]-c.iloc[j,i])*x[i,j]+c.iloc[j,i] for i in range(n-1) for j in range(i+1,n))==orig_k)                
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
        print('Start pair optimization')
    tic = time.perf_counter()
    AP.optimize()
    toc = time.perf_counter()
    if verbose:
        print(f"Optimization in {toc - tic:0.4f} seconds")
        print('End optimization')

    sol_x = get_sol_x_by_x(x,n)()
    sol_u = get_sol_x_by_x(u,n)()
    sol_v = get_sol_x_by_x(v,n)()
    r = np.sum(sol_x,axis=0)
    ranking = np.argsort(r)
    perm = tuple([int(item) for item in ranking])
    #perm = tuple(perm_inxs[np.array(key)])
    #reorder = np.argsort(perm_inxs)
    if method == 'lop':
        k = np.sum(np.sum(D*sol_x))
    elif method == 'hillside':
        k = np.sum(np.sum(c*sol_x))
        
    details = {"obj":k,"perm":perm,"x":sol_x,"u":sol_u,"v":sol_v}
    return AP.objVal,details

def bilp_max_tau_jonad(D,lazy=False,verbose=True):
    first_k, first_details = solve(D,method='lop',lazy=lazy,verbose=verbose)
    if verbose:
        print('Finished first optimization')
    n = D.shape[0]
        
    AP = Model('lop')

    x = {}
    y = {}
    z = {}
    
    for i in range(n):
        for j in range(n):
            x[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="x(%s,%s)"%(i,j)) #binary
            y[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="y(%s,%s)"%(i,j)) #binary
            z[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="b(%s,%s)"%(i,j)) #binary
            
    
    AP.update()
    
    for i in range(n):
        AP.addConstr(z[i,i]==0)
    
    for i in range(n):
            for j in range(i+1,n):         
                    AP.addConstr(x[i,j] + x[j,i] == 1)
                    AP.addConstr(y[i,j] + y[j,i] == 1)
    
    
    for i in range(n-1):
        for j in range(i+1,n):
            for k in range(i+1,n):
                if k!=j:
                    trans_cons = []
                    trans_cons.append(AP.addConstr(x[i,j] + x[j,k] + x[k,i] <= 2))
                    trans_cons.append(AP.addConstr(y[i,j] + y[j,k] + y[k,i] <= 2))
                    if lazy:
                        for cons in trans_cons:
                            cons.setAttr(GRB.Attr.Lazy,1)
    AP.update()
    AP.addConstr(quicksum((D[i,j])*x[i,j] for i in range(n) for j in range(n)) == first_k)
    AP.addConstr(quicksum((D[i,j])*y[i,j] for i in range(n) for j in range(n)) == first_k)

    AP.update()
    for i in range(n):
        for j in range(n):
            if i != j:
                AP.addConstr(x[i,j]+y[i,j]-z[i,j] <= 1)
           
    AP.update()

    AP.setObjective(quicksum((z[i,j]) for i in range(n) for j in range(n)),GRB.MINIMIZE)
    AP.setParam( 'OutputFlag', verbose )
    AP.update()
        
    if verbose:
        print('Start optimization')
    tic = time.perf_counter()
    AP.update()
    AP.optimize()
    toc = time.perf_counter()
    if verbose:
        print(f"Optimization in {toc - tic:0.4f} seconds")
        print('End optimization')
    
    sol_x = get_sol_x_by_x(x,n)()
    sol_y = get_sol_x_by_x(y,n)()
    sol_z = get_sol_x_by_x(z,n)()
    r = np.sum(sol_x,axis=0)
    ranking = np.argsort(r)
    perm_x = tuple([int(item) for item in ranking])
    
    r = np.sum(sol_y,axis=0)
    ranking = np.argsort(r)
    perm_y = tuple([int(item) for item in ranking])
    
    k_x = np.sum(D*sol_x)
    k_y = np.sum(D*sol_y)
    
    details = {"obj":AP.objVal,"k_x": k_x, "k_y":k_y, "perm_x":perm_x,"perm_y":perm_y, "x": sol_x,"y":sol_y,"z":sol_z}
            
    return first_k,details

def solve_pair_min_tau(D,D2=None,method=["lop","hillside"][1],lazy=False,verbose=True,min_dis=1):
    first_k, first_details = solve(D,method=method,lazy=lazy,verbose=verbose)
    if verbose:
        print('Finished first optimization. Obj:',first_k)
    n = D.shape[0]
    if D2 is not None:
        assert n == D2.shape[0]
        
    AP = Model(method)
    
    second_k = first_k
    if D2 is not None:
        second_k, second_details = solve(D2,method=method,lazy=lazy,verbose=verbose)
    
    if method == 'lop':
        c1 = D
        c2 = D
        if D2 is not None:
            c2 = D2
    elif method == 'hillside':
        c1 = C_count(D)
        c2 = c1
        if D2 is not None:
            c2 = C_count(D2)

    x = {}
    y = {}
    u = {}
    v = {}
    b = {}
    for i in range(n-1):
        for j in range(i+1,n):
            x[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="x(%s,%s)"%(i,j)) #binary
            y[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="y(%s,%s)"%(i,j)) #binary
            u[i,j] = AP.addVar(name="u(%s,%s)"%(i,j),vtype=GRB.BINARY,lb=0,ub=1) #nonnegative
            v[i,j] = AP.addVar(name="v(%s,%s)"%(i,j),vtype=GRB.BINARY,lb=0,ub=1) #nonnegative
    AP.update()
    
    for i in range(n-1):
        for j in range(i+1,n):
            for k in range(j+1,n):
                trans_cons = []
                trans_cons.append(AP.addConstr(x[i,j] + x[j,k] - x[i,k] <= 1))
                trans_cons.append(AP.addConstr(x[i,j] + x[j,k] - x[i,k] >= 0))
                trans_cons.append(AP.addConstr(y[i,j] + y[j,k] - y[i,k] <= 1))
                trans_cons.append(AP.addConstr(y[i,j] + y[j,k] - y[i,k] >= 0))
                if lazy:
                    for cons in trans_cons:
                        cons.setAttr(GRB.Attr.Lazy,1)
    AP.update()
    
    AP.addConstr(quicksum((c1.iloc[i,j]-c1.iloc[j,i])*x[i,j]+c1.iloc[j,i] for i in range(n-1) for j in range(i+1,n)) == first_k)
    AP.addConstr(quicksum((c2.iloc[i,j]-c2.iloc[j,i])*y[i,j]+c2.iloc[j,i] for i in range(n-1) for j in range(i+1,n)) == second_k)

    AP.update()
    for i in range(n-1):
        for j in range(i+1,n):
            AP.addConstr(u[i,j] - v[i,j] == x[i,j] - y[i,j])
            AP.addConstr(u[i,j] + v[i,j] <= 1)
    AP.update()
    
    AP.addConstr(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)) >= min_dis)    
    AP.update()

    AP.setObjective(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)),GRB.MINIMIZE)
    AP.setParam( 'OutputFlag', verbose )
    AP.update()
        
    if verbose:
        print('Start optimization')
    tic = time.perf_counter()
    AP.update()
    AP.optimize()
    toc = time.perf_counter()
    if verbose:
        print(f"Optimization in {toc - tic:0.4f} seconds")
        print('End optimization')
    
    sol_x = get_sol_x_by_x(x,n)()
    sol_y = get_sol_x_by_x(y,n)()
    sol_v = get_sol_uv_by_x(v,n)()
    sol_u = get_sol_uv_by_x(u,n)()
    r = np.sum(sol_x,axis=0)
    ranking = np.argsort(r)
    perm_x = tuple([int(item) for item in ranking])
    
    r = np.sum(sol_y,axis=0)
    ranking = np.argsort(r)
    perm_y = tuple([int(item) for item in ranking])
    
    k_x = np.sum(np.sum(c1*sol_x))
    k_y = np.sum(np.sum(c2*sol_y))
    
    details = {"obj":AP.objVal,"k_x": k_x, "k_y":k_y, "perm_x":perm_x,"perm_y":perm_y, "x": sol_x,"y":sol_y,"u":sol_u,"v":sol_v}
            
    return AP.objVal,details
    
def solve_pair_max_tau(D,D2=None,method=["lop","hillside"][1],lazy=False,verbose=True,cont=False):
    _, first_details = solve(D,method=method,lazy=lazy,verbose=verbose,cont=cont)
    first_k = first_details['obj']
    if verbose:
        print('Finished first optimization. Obj:',first_k)
    n = D.shape[0]
    if D2 is not None:
        assert n == D2.shape[0]
        
    AP = Model(method)
    
    second_k = first_k
    if D2 is not None:
        _, second_details = solve(D2,method=method,lazy=lazy,verbose=verbose,cont=cont)
        second_k = second_details['obj']
    
    if method == 'lop':
        c1 = D
        c2 = D
        if D2 is not None:
            c2 = D2
    elif method == 'hillside':
        c1 = C_count(D)
        c2 = c1
        if D2 is not None:
            c2 = C_count(D2)

    x = {}
    y = {}
    u = {}
    v = {}
    b = {}
    for i in range(n-1):
        for j in range(i+1,n):
            x[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="x(%s,%s)"%(i,j)) #binary
            y[i,j] = AP.addVar(lb=0,vtype=GRB.BINARY,ub=1,name="y(%s,%s)"%(i,j)) #binary
            u[i,j] = AP.addVar(name="u(%s,%s)"%(i,j),vtype=GRB.BINARY,lb=0,ub=1) #nonnegative
            v[i,j] = AP.addVar(name="v(%s,%s)"%(i,j),vtype=GRB.BINARY,lb=0,ub=1) #nonnegative
    AP.update()
    
    for i in range(n-1):
        for j in range(i+1,n):
            for k in range(j+1,n):
                trans_cons = []
                trans_cons.append(AP.addConstr(x[i,j] + x[j,k] - x[i,k] <= 1))
                trans_cons.append(AP.addConstr(x[i,j] + x[j,k] - x[i,k] >= 0))
                trans_cons.append(AP.addConstr(y[i,j] + y[j,k] - y[i,k] <= 1))
                trans_cons.append(AP.addConstr(y[i,j] + y[j,k] - y[i,k] >= 0))
                if lazy:
                    for cons in trans_cons:
                        cons.setAttr(GRB.Attr.Lazy,1)
    AP.update()
    
    AP.addConstr(quicksum((c1.iloc[i,j]-c1.iloc[j,i])*x[i,j]+c1.iloc[j,i] for i in range(n-1) for j in range(i+1,n)) == first_k)
    AP.addConstr(quicksum((c2.iloc[i,j]-c2.iloc[j,i])*y[i,j]+c2.iloc[j,i] for i in range(n-1) for j in range(i+1,n)) == second_k)

    AP.update()
    for i in range(n-1):
        for j in range(i+1,n):
            AP.addConstr(u[i,j] - v[i,j] == x[i,j] - y[i,j])
            AP.addConstr(u[i,j] + v[i,j] <= 1)
    AP.update()

    AP.setObjective(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)),GRB.MAXIMIZE)
    AP.setParam( 'OutputFlag', verbose )
    AP.update()
        
    if verbose:
        print('Start optimization')
    tic = time.perf_counter()
    AP.update()
    AP.optimize()
    toc = time.perf_counter()
    if verbose:
        print(f"Optimization in {toc - tic:0.4f} seconds")
        print('End optimization')
    
    sol_x = get_sol_x_by_x(x,n)()
    sol_y = get_sol_x_by_x(y,n)()
    sol_v = get_sol_uv_by_x(v,n)()
    sol_u = get_sol_uv_by_x(u,n)()
    r = np.sum(sol_x,axis=0)
    ranking = np.argsort(r)
    perm_x = tuple([int(item) for item in ranking])
    
    r = np.sum(sol_y,axis=0)
    ranking = np.argsort(r)
    perm_y = tuple([int(item) for item in ranking])
    
    k_x = np.sum(np.sum(c1*sol_x))
    k_y = np.sum(np.sum(c2*sol_y))
    
    details = {"obj":AP.objVal,"k_x": k_x, "k_y":k_y, "perm_x":perm_x,"perm_y":perm_y, 'c1': c1, 'c2': c2, "x": sol_x,"y":sol_y,"u":sol_u,"v":sol_v}
            
    return AP.objVal,details
    