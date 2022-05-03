import numpy as np

def banded_matrix(N):
    arr = np.zeros((N,N))
    for d in range(-N, N):
        arr += np.diag(np.repeat(abs(d), N - abs(d)), d)
    return np.matrix(arr)

def weighted_matrix(N):
    return np.matrix([[1 / i for _ in range(1, N + 1)] for i in range(1, N + 1)])

def beta(Xstar_r_r, normalize = True, average=False):
    Xstar_r_r = Xstar_r_r.copy()
    mask = ((Xstar_r_r.values > 0) & (Xstar_r_r.values < 1))
    num_frac = sum(sum(mask))
    Xstar_r_r.values[:,:] = mask.astype(int)
    n = len(Xstar_r_r)
    worst_case_Xstar_r_r = np.ones(Xstar_r_r.shape)
    def _beta(Xstar_r_r,n):
        return (Xstar_r_r * banded_matrix(n) * weighted_matrix(n)).sum().sum()
    result = _beta(Xstar_r_r,n)
    if normalize:
        result = result/_beta(worst_case_Xstar_r_r,n)
    if average and num_frac > 0:
        result = result/num_frac
    return result

def calc_beta(D):
    obj,details = pyrankability.rank.solve(D,method="lop",cont=True)
    Xstar = pd.DataFrame(pyrankability.common.threshold_x(details['x']),index=D.index,columns=D.columns)
    perm = details['P'][0] # select one permutation
    Xstar_r_r = Xstar.iloc[np.array(perm),np.array(perm)]
    return pyrankability.features.beta(Xstar_r_r)

def calc_nmos(D,max_num_solutions=1000):
    obj_lop_scip,details_lop_scip = pyrankability.rank.solve(D,method="lop",include_model=True,cont=False)
    model = details_lop_scip['model']
    model_file = pyrankability.common.write_model(model)
    max_num_solutions = 1000
    results = pyrankability.search.scip_collect(D,model_file,max_num_solutions=max_num_solutions) 
    return len(results['perms'])
