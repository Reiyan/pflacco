import os
import numpy as np
import pandas as pd
import pytest
from pflacco.sampling import create_initial_sample

RSC = os.path.join('tests', 'resources')


@pytest.mark.parametrize('dim', [2, 5, 10])
def test_d2_sample(dim):
    np.random.seed(50)
    sample = create_initial_sample(dim, lower_bound = -5, upper_bound = 5)
    expected = pd.read_pickle(os.path.join(RSC, f'regular_sample_d{dim}.pkl'))
    assert sample.equals(expected)

def test_different_bounds_sample():
    np.random.seed(50)
    sample = create_initial_sample(5, lower_bound = [-1, 3, 5, 2, 1], upper_bound = 10)
    expected = pd.read_pickle(os.path.join(RSC, f'bound_sample.pkl'))
    assert sample.equals(expected)

def test_sobol_sample():
    sample = create_initial_sample(2, sample_type = 'sobol', seed = 50)
    expected = pd.read_pickle(os.path.join(RSC, f'sobol_sample.pkl'))
    assert sample.equals(expected)

def test_random_sample():
    np.random.seed(50)
    sample = create_initial_sample(5, sample_type = 'random')
    expected = pd.read_pickle(os.path.join(RSC, f'random_sample.pkl'))
    assert sample.equals(expected)



