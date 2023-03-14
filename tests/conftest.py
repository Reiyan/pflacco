import os
import pandas as pd
import pytest

RSC = os.path.join('tests', 'resources')

@pytest.fixture(scope="package")
def x_samples():
    X = pd.read_pickle(os.path.join(RSC, 'init_sample.pkl'))
    return X
