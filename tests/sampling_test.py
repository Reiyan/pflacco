import os
import numpy as np
import pandas as pd
import pytest
from pflacco.sampling import create_initial_sample

RSC = os.path.join('tests', 'resources')

@pytest.mark.parametrize('dim', [2, 5, 10])
def test_d2_sample(dim):
    sample = create_initial_sample(dim, lower_bound = -5, upper_bound = 5, seed = 50)
    expected = pd.read_pickle(os.path.join(RSC, f'regular_sample_d{dim}.pkl'))
    assert sample.equals(expected)

def test_different_bounds_sample():
    sample = create_initial_sample(5, lower_bound = [-1, 3, 5, 2, 1], upper_bound = 10, seed = 50)
    expected = pd.read_pickle(os.path.join(RSC, f'bound_sample.pkl'))
    assert sample.equals(expected)

def test_sobol_sample():
    sample = create_initial_sample(2, sample_type = 'sobol', seed = 50)
    expected = pd.read_pickle(os.path.join(RSC, f'sobol_sample.pkl'))
    assert sample.equals(expected)

def test_random_sample():
    sample = create_initial_sample(5, sample_type = 'random', seed = 50)
    expected = pd.read_pickle(os.path.join(RSC, f'random_sample.pkl'))
    assert sample.equals(expected)

def test_random_mixed_search_space_sample(x_mixed_search_space_sample):
    sample = create_initial_sample(3, sample_coefficient= 3, sample_type = 'random', categorical_values = ['cont', 'int', ['val1', 'val2', 'val3']], seed = 50)
    assert sample.equals(x_mixed_search_space_sample)

def test_random_mixed_search_space_sample_not_implemented():
    with pytest.raises(NotImplementedError) as e:  
        sample = create_initial_sample(2, sample_type = 'lhs', categorical_values = [None, ['val1', 'val2', 'val3']])
    assert str(e.value) == 'Currently, only "random" sampling is enabled for mixed search spaces.'

# https://github.com/Reiyan/pflacco/issues/42
def test_sobol_sample_honours_seed():
    kwargs = dict(sample_type = 'sobol')
    assert create_initial_sample(2, 8, seed = 1, **kwargs).equals(create_initial_sample(2, 8, seed = 1, **kwargs))
    assert not create_initial_sample(2, 8, seed = 1, **kwargs).equals(create_initial_sample(2, 8, seed = 2, **kwargs))

# https://github.com/Reiyan/pflacco/issues/31
@pytest.mark.parametrize('sample_type', ['lhs', 'sobol', 'random'])
def test_seed_accepts_an_int_or_a_generator(sample_type):
    kwargs = dict(dim = 2, n = 8, sample_type = sample_type)
    # the same int reproduces, a different one does not
    assert create_initial_sample(seed = 1, **kwargs).equals(create_initial_sample(seed = 1, **kwargs))
    assert not create_initial_sample(seed = 1, **kwargs).equals(create_initial_sample(seed = 2, **kwargs))
    # a Generator is accepted in place of the int and behaves identically
    assert create_initial_sample(seed = np.random.default_rng(1), **kwargs).equals(
           create_initial_sample(seed = np.random.default_rng(1), **kwargs))

