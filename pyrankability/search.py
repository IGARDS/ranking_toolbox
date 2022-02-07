# -*- coding: utf-8 -*-
import itertools
import copy
import multiprocessing
import tempfile
import os
import shutil
import time
import subprocess

import numpy as np
from gurobipy import *
from joblib import Parallel, delayed

from .rank import *

from . import common

def solve_any_diff(D,orig_obj,orig_sol_x,method=["lop","hillside"][1],lazy=False,verbose=False) :
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

def solve_fixed_binary_x(D,orig_k,orig_sol_x,minimize=False,method=["lop","hillside"][1],lazy=False,verbose=False) :
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
            u[i,j] = AP.addVar(name="u(%s,%s)"%(i,j),vtype=GRB.BINARY)
            v[i,j] = AP.addVar(name="v(%s,%s)"%(i,j),vtype=GRB.BINARY)
    AP.update()
    for i in range(n-1):
        for j in range(i+1,n):
            AP.addConstr(u[i,j] - v[i,j] == x[i,j] - orig_sol_x[i,j])
            AP.addConstr(u[i,j] + v[i,j] <= 1)
    AP.update()

    if not minimize:
        AP.setObjective(quicksum(u[i,j]+v[i,j] for i in range(n-1) for j in range(i+1,n)),GRB.MAXIMIZE)
    else:
        AP.setObjective(quicksum(u[i,j]+v[i,j] for i in range(n-1) for j in range(i+1,n)),GRB.MINIMIZE)
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
    details = {"obj":AP.objVal,"perm":perm,"x":sol_x,"sol_u":sol_u,"sol_v":sol_v}
    if method == 'hillside':
        details['c'] = c
        k = round(k)
    elif method == 'lop': # switch to delta
        Dre = D.values[perm,:][:,perm]
        #print(k,np.sum(np.triu(Dre)))
        k = np.sum(np.tril(Dre,k=-1))
        
    return k,details



def solve_fixed_cont_x(D,orig_k,orig_sol_x,minimize=False,method=["lop","hillside"][1],lazy=False,verbose=False) :
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

    if not minimize:
        AP.setObjective(quicksum(u[i,j]+v[i,j] for i in range(n-1) for j in range(i+1,n)),GRB.MAXIMIZE)
    else:
        AP.setObjective(quicksum(u[i,j]+v[i,j] for i in range(n-1) for j in range(i+1,n)),GRB.MINIMIZE)
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
    details = {"obj":AP.objVal,"perm":perm,"x":sol_x,"u":sol_u,"v":sol_v}
    if method == 'hillside':
        details['c'] = c
        k = round(k)
    elif method == 'lop': # switch to delta
        Dre = D.values[perm,:][:,perm]
        #print(k,np.sum(np.triu(Dre)))
        k = np.sum(np.tril(Dre,k=-1))
        
    return k,details

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

"""
    if tau_range is not None:
        ndis_thres1 = common.tau_to_ndis(tau_range[0],len(D))
        ndis_thres2 = common.tau_to_ndis(tau_range[1],len(D))
"""
def solve_pair(D,D2=None,method=["lop","hillside"][1],minimize=False,min_ndis=None,max_ndis=None,tau_range=None,lazy=False,verbose=False):
    if tau_range is not None:
        ndis_thres1 = common.tau_to_ndis(tau_range[0],len(D))
        ndis_thres2 = common.tau_to_ndis(tau_range[1],len(D)) 
        
    #import pdb; pdb.set_trace()
    delta1, first_details = solve(D,method=method,lazy=lazy,verbose=verbose)
    first_k = first_details['obj']
    if verbose:
        print('Finished first optimization. Obj:',first_k)
        
    n = D.shape[0]
    if D2 is not None:
        assert n == D2.shape[0]
        
    AP = Model(method)
    
    second_k = first_k
    if D2 is not None:
        delta2, second_details = solve(D2,method=method,lazy=lazy,verbose=verbose)
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
    
    if min_ndis is not None:
        AP.addConstr(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)) >= min_ndis)    
    if max_ndis is not None:
        AP.addConstr(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)) <= max_ndis)    
    
    if tau_range is not None:
        AP.addConstr(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)) >= ndis_thres2 )  
        AP.addConstr(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)) <= ndis_thres1 )
    AP.update()

    if minimize:
        AP.setObjective(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)),GRB.MINIMIZE)
    else:
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
    
    details = {"obj":AP.objVal,"k_x": k_x, "k_y":k_y, "perm_x":perm_x,"perm_y":perm_y, "x": sol_x,"y":sol_y,"u":sol_u,"v":sol_v}
            
    return AP.objVal,details
    
def solve_pair_min_tau(D,D2=None,method=["lop","hillside"][1],lazy=False,verbose=True,cont=False,tau_range=None):
    if tau_range is not None:
        ndis_thres1 = common.tau_to_ndis(tau_range[0],len(D))
        ndis_thres2 = common.tau_to_ndis(tau_range[1],len(D))
    
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
    
    if tau_range is not None:
        AP.addConstr(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)) >= ndis_thres2 )  
        AP.addConstr(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)) <= ndis_thres1 )  
        AP.update()
        
    if verbose:
        print('Start optimization')
    AP.params.Threads=7
    AP.update()
    if cont:
        AP.Params.Method = 2
        AP.Params.Crossover = 0    
        AP.update()
    tic = time.perf_counter()
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
    tau = common.tau(perm_x,perm_y)
    ncon,ndis = common.calc_con_dis(perm_x,perm_y)
    
    details = {"obj":AP.objVal,"tau":tau,"ncon":ncon,"ndis":ndis,"k_x": k_x, "k_y":k_y, "perm_x":perm_x,"perm_y":perm_y, 'c1': c1, 'c2': c2, "x": sol_x,"y":sol_y,"u":sol_u,"v":sol_v}
            
    return AP.objVal,details
    
def solve_pair_tau_range(tau_range,D,D2=None,method=["lop","hillside"][1],lazy=False,verbose=True,cont=False):
    ndis_thres1 = common.tau_to_ndis(tau_range[0],len(D))
    ndis_thres2 = common.tau_to_ndis(tau_range[1],len(D))
    
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
    
    AP.addConstr(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)) >= ndis_thres2 )  
    AP.addConstr(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)) <= ndis_thres1 )  
    AP.update()

    #AP.setObjective(quicksum((u[i,j]+v[i,j]) for i in range(n-1) for j in range(i+1,n)),GRB.MINIMIZE)
    AP.setParam( 'OutputFlag', verbose )
    AP.update()
        
    if verbose:
        print('Start optimization')
    AP.params.Threads=7
    AP.update()
    if cont:
        AP.Params.Method = 2
        AP.Params.Crossover = 0    
        AP.update()
    tic = time.perf_counter()
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
    
    tau = common.tau(perm_x,perm_y)
    ncon,ndis = common.calc_con_dis(perm_x,perm_y)
    
    details = {"obj":AP.objVal,"tau":tau,"ncon":ncon,"ndis":ndis,"k_x": k_x, "k_y":k_y, "perm_x":perm_x,"perm_y":perm_y, 'c1': c1, 'c2': c2, "x": sol_x,"y":sol_y,"u":sol_u,"v":sol_v}
            
    return AP.objVal,details

def collect(D_or_C,model,opt_k):
    #model = details_lop_with_models['model']
    model_file = common.write_model(model)
    solution_file = model_file + ".solutions"
    r1 = os.system(f"sed -i '/^OBJSENS/d' {model_file}")
    if r1 != 0:
        raise Exception("Unknown error in [1]")
    this_dir = os.path.dirname(os.path.abspath(__file__))
    r2 = os.system(f'{this_dir}/../collect.sh {model_file} {solution_file}')
    if r2 != 0:
        raise Exception("Unknown error in [2]")

    solutions = pd.read_csv(solution_file,sep=', ')
    x_columns = solutions.columns[1:-1]
    xs = []
    a,b,c = 1,1,-2*len(x_columns)
    n = int((-b + np.sqrt(b**2 - 4*a*c))/(2*a) + 1)
    xstar = np.zeros((n,n))
    objs = []
    s = 0
    for k in range(solutions.shape[0]):
        x = np.zeros((n,n))
        for c in x_columns:
            ij_str = c.replace("x(","").replace(")","")
            i,j = ij_str.split(",")
            i,j = int(i),int(j)
            x[i,j] = solutions.loc[k,c]
            x[j,i] = 1 - x[i,j]
        obj = np.sum(np.sum(D_or_C*x))
        xs.append(x)
        objs.append(obj)
        error = obj - opt_k
        xstar += x
    xstar = xstar/solutions.shape[0]

    perms = []
    for x in xs:
        r = np.sum(x,axis=0)
        perm = np.argsort(r)
        perms.append(perm)
        
    return perms, xs, xstar

def scip_collect(D,model_file,max_num_solutions=1000,solution_file=common.get_temp_model_solution(),show_output=True,compute_C_func=None):
    assert type(D) == pd.DataFrame
    assert D.shape[0] == D.shape[1]
    assert set(D.index) == set(D.columns)
    items = D.index
    # Make sure the model file is in the correct format
    r = os.system(f"sed '/^OBJSENS/d' {model_file} > {model_file}.fixed.mps")
    if r != 0:
        raise Exception(f"Unknown error while trying to fix {model_file}")
    
    #set constraints countsols sollimit 1000
    max_num_solutions = str(max_num_solutions)
    cmd = ["scip_collect.sh", f"{model_file}.fixed.mps", solution_file,max_num_solutions]
    if show_output:
        print("Running:"," ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True)
    if show_output:
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
              
    solutions = pd.read_csv(solution_file,sep=', ')
    x_columns = solutions.columns[1:-1]
    xs = []
    perms = []
    rankings = []
    a,b,c = 1,1,-2*len(x_columns)
    n = int((-b + np.sqrt(b**2 - 4*a*c))/(2*a) + 1)
    xstar = np.zeros((n,n))
              
    if compute_C_func is not None:
        C = compute_C_func(D)
    else:
        C = D
    objs = []
    s = 0
    for k in range(solutions.shape[0]):
        x = np.zeros((n,n))
        for c in x_columns:
            ij_str = c.replace("x(","").replace(")","")
            i,j = ij_str.split(",")
            i,j = int(i),int(j)
            x[i,j] = solutions.loc[k,c]
            x[j,i] = 1 - x[i,j]
        obj = np.sum(np.sum(C*x))
        xs.append(x)
        objs.append(obj)
        xstar += x
        r = np.sum(x,axis=0)
        ranking = pd.Series(r,index=items)
        rankings.append(ranking)
        perm = tuple([int(item) for item in np.argsort(r)])
        perms.append(perm)
    xstar = xstar/solutions.shape[0]
    xstar = pd.DataFrame(xstar,index=items,columns=items)
    results = {"xs":xs, "objs":objs,"xstar":xstar,"perms":perms,"rankings":rankings}
    return results

def scip_count():
    pass