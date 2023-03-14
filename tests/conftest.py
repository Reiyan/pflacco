import os
import pandas as pd
import pytest

RSC = os.path.join('tests', 'resources')

@pytest.fixture(scope="package")
def x_samples():
    X = pd.read_csv(os.path.join(RSC, 'init_sample.csv'))
    return X

#@pytest.fixture
#def ela_classical_feature_values():
#    X = pd.read_csv(os.path.join(RSC, 'test_classical_ela_features.csv'))
#    return X

