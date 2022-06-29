import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import numpy as np
import pandas as pd

# Import the student solutions
import pyrankability

RPLIB_DATA_PREFIX=f"{DIR}../data"

def test_1():
    n=8
    D=np.zeros((n,n))
    D[np.triu_indices(n,1)]=1
    D[[5,3,7]] = 1-D[[5,3,7]]
    D=pd.DataFrame(D)
    k_hillside,details_hillside = pyrankability.rank.solve(D,method='hillside')
    k_lop,details_lop = pyrankability.rank.solve(D,method='lop')

    assert k_hillside == 54 and k_lop == 12.0
