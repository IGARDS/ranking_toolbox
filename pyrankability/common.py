import math
import numpy as np
import json
import sys
import string
import random
import tempfile
import os
import copy

np.set_printoptions(threshold=sys.maxsize)

def get_temp_model_path():
    temp_dir = tempfile.mkdtemp()
    model_file = os.path.join(temp_dir,"model.mps")
    return model_file

def write_model(AP,model_file=None):
    if model_file is None:
        model_file = get_temp_model_path()
    AP.write(model_file)
    return model_file

"""
def compute_C(D):
    D = copy.copy(D)
    for i in range(D.shape[0]):
        D[i,i] = np.nan
    c = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            c[i,j] = np.count_nonzero(D[:,j]-D[:,i]<0) + np.count_nonzero(D[i,:]-D[j,:]<0) 
    return c
"""

# I'm not sure this works exactly the same for LOP
def perm_to_x(perm):
    x = np.triu(np.ones((len(perm),len(perm))),1)
    order = np.argsort(np.array(perm))
    x = 1-x[np.ix_(order,order)]
    return x

def get_sol_x_by_x(x,n,cont=False):
    f = int
    default = 0
    if cont:
        f = float
        default = 0.0
    def myfunc():
        values = []
        for i in range(n):
            for j in range(n):
                if i==j:
                    values.append(default)
                elif (i,j) in x:
                    if type(x[i,j]) == int or type(x[i,j]) == float:
                        values.append(f(x[i,j]))
                    else:
                        values.append(f(x[i,j].X))
                elif (j,i) in x:
                    if type(x[j,i]) == int or type(x[j,i]) == float:
                        values.append(f(1-x[j,i]))                    
                    else:
                        values.append(f(1-x[j,i].X))                    
                elif i < j: # don't care about this pair
                    values.append(default)
                else:
                    values.append(1-default)
        return np.reshape(values,(n,n))
    return myfunc

def get_sol_uv_by_x(x,n):
    def myfunc():
        values = []
        for i in range(n):
            for j in range(n):
                if (i,j) in x:
                    values.append(int(x[i,j].X))
                else:
                    if i==j:
                        values.append(0)
                    elif i < j:
                        values.append(int(x[i,j].X))
                    else:
                        values.append(0)
        return np.reshape(values,(n,n))
    return myfunc

def threshold_x(x,lower_cut=1e-3,upper_cut=1-1e-3):
    x = x.copy()
    cut_ixs = np.where(x < lower_cut)
    x[cut_ixs] = 0.
    cut_ixs = np.where(x > upper_cut)
    x[cut_ixs] = 1.
    return x

def compare_objective_values(o1,o2,tol=1**-6):
    if abs(o1-o2) <= tol:
        return True
    return False

def random_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

def round_Xn(Xn,mult=10):
    return round(mult*Xn)*1./mult

import math

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def calc_con_dis(perm_x,perm_y):
    ncon = 0
    ndis = 0
    for i in range(len(perm_x)): # i is an item
        i_x = np.where(np.array(perm_x)==i)[0] # location of i in perm_x
        i_y = np.where(np.array(perm_y)==i)[0] # location of i in perm_y
        for j in range(i+1,len(perm_x)): # j is another item
            j_x = np.where(np.array(perm_x)==j)[0] # location of j in perm_x
            j_y = np.where(np.array(perm_y)==j)[0] # location of j in perm_y
            if i_x-j_x > 0 and i_y-j_y < 0:
                ndis += 1
            elif i_x-j_x < 0 and i_y-j_y > 0:
                ndis += 1
            elif i_x-j_x > 0 and i_y-j_y > 0:
                ncon += 1
            elif i_x-j_x < 0 and i_y-j_y < 0:
                ncon += 1
    return ncon, ndis

def tau_to_ndis(tau_val,n):
    t = nCr(n,2)
    ndis = (tau_val*t - t)/-2
    return int(ndis)

def tau(perm_x,perm_y):
    n = len(perm_x)
    ncon,ndis = calc_con_dis(perm_x,perm_y)
    return tau_from_ncon_ndis(ncon,ndis,n)
    
def tau_from_ncon_ndis(ncon,ndis,n):
    return (ncon - ndis)/nCr(n,2)

def nPr(n,r):
    f = math.factorial
    return f(n) // f(n-r)

def permute_D(D,perm):
    return D[perm,:][:,perm]

def calc_k(D,max_value=None):
    if not max_value:
        max_value = np.max(D)
        if max_value == 0:
            max_value = 1
    perfectRG=np.triu(max_value*np.ones((D.shape[0],D.shape[0])),1).astype(int)
    k = np.sum(np.abs(perfectRG-D))
    return k

def define_D(M,w,min_support):
    n = M.shape[0]
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = 0
            for k in range(len(w)):
                c += w[k]*1.0*int(M[i,k] > M[j,k])
            D[i,j] = int(c > min_support)
            
    return D

def len_chunks(l,n):
    return len(range(0,len(l),n))

def chunks(l, n):
    chunks_ = []
    for i in range(0, len(l), n):
        #yield l[i:i + n]
        chunks_.append(l[i:i + n])
    return chunks_

def chunks_generator(l,n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def as_json(k,P,other={},p=None):
    if len(P) > 0 and np.min(P) == 0:
        P = (np.array(P,dtype=int)+1).tolist()
    if p == None:
        p = len(P)
    return json.dumps({"k":k,"p": p, "P":P,"other":other})

def json_string(k,P,other={},p=None):
    p = len(P)
    if len(P) > 0 and np.min(P) == 0:
        P = (np.array(P,dtype=int)+1).tolist()
    if p == None:
        p = len(P)
    k = int(k)
    indent = "    "
    instance_as_string = "{\n"
    instance_as_string += indent + json.dumps({"k": k}).replace("{","").replace("}","")+",\n"
    instance_as_string += indent + json.dumps({"p": p}).replace("{","").replace("}","")+",\n"
    instance_as_string += indent + '"P": \n'
    P_as_string = np.array2string(np.array(P,dtype=int),separator=",",max_line_width=np.Inf).replace("[","[\n",1)
    lines = P_as_string.split("\n")
    for i,line in enumerate(lines):
        add_indent = indent+indent
        if i > 0:
            add_indent += indent
        instance_as_string += add_indent + line + "\n"
    instance_as_string = instance_as_string[:-2] + "],\n"
    instance_as_string += indent + json.dumps({"other": other})[1:-1]+"\n"
    instance_as_string += "}"
    return instance_as_string


class Search:
    def to_json(self):
        solution = {}
        solution["k"] = int(self.k)
        solution["p"] = len(self.P)
        solution["P"] = [[int(v) + 1 for v in perm] for perm in self.P]
        solution["other"] = self.other_to_dict()
        return solution
    
    # Override this if the search subclass does not keep track of thse things
    def other_to_dict(self):
        return {"skipped": self.skipped, "searched": self.searched, "exact_solution_found": self.exact_solution_found}

