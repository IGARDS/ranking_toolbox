import numpy as np

def banded_matrix(N):
  arr = np.zeros((N,N))
  for d in range(-N, N):
    arr += np.diag(np.repeat(abs(d), N - abs(d)), d)
  return np.matrix(arr)

def weighted_matrix(N):
  return np.matrix([[1 / i for _ in range(1, N + 1)] for i in range(1, N + 1)])

def beta(Xstar_r_r, normalize = True):
  Xstar_r_r = Xstar_r_r.copy()
  Xstar_r_r.values[:,:] = np.ceil(Xstar_r_r.values)
  n = len(Xstar_r_r)
  worst_case_Xstar_r_r = np.ones(Xstar_r_r.shape)
  def _beta(Xstar_r_r,n):
    return (Xstar_r_r * banded_matrix(n) * weighted_matrix(n)).sum().sum()
  if normalize == True:
    return _beta(Xstar_r_r,n)/_beta(worst_case_Xstar_r_r,n)
  else:
    return _beta(Xstar_r_r,n)