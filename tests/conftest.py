import os
import pandas as pd
import pytest

RSC = os.path.join('tests', 'resources')

@pytest.fixture(scope="package")
def x_samples():
    X = pd.read_pickle(os.path.join(RSC, 'init_sample.pkl'))
    return X

@pytest.fixture(scope="package")
def x_mixed_search_space_sample():
    X = [['0.49460164553802144', '0', 'val3'], 
         ['0.39632990972277693', '1', 'val1'], 
         ['0.40819720033319706', '0', 'val2'], 
         ['0.31000934868533203', '0', 'val2'], 
         ['0.14546685649615498', '1', 'val3'], 
         ['0.559957103722125', '1', 'val1'], 
         ['0.674573065822264', '0', 'val3'], 
         ['0.524103502664114', '0', 'val3'], 
         ['0.66833756877668', '0', 'val1']]
    X = pd.DataFrame(X, columns = ['x0', 'x1', 'x2'])
    return X