import os
import pandas as pd
import pytest

from ioh import get_problem
from pflacco.misc_features import *

DIMS = [2, 5, 10]
RSC = os.path.join('tests', 'resources')

@pytest.fixture
def feature_values():
    X = pd.read_pickle(os.path.join(RSC, 'test_misc_ela_features.pkl'))
    return X

def test_calculate_fitness_distance_correlation(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_fitness_distance_correlation(tmp, y)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert result[colnames].equals(feature_values[colnames])

def test_calculate_hill_climbing_features(feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            f = get_problem(fid, 1, dim)
            features = calculate_hill_climbing_features(f, dim, lower_bound = -5, upper_bound = 5, seed = 200)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert result[colnames].equals(feature_values[colnames])

def test_calculate_gradient_features(feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            f = get_problem(fid, 1, dim)
            features = calculate_gradient_features(f, dim, lower_bound=-5, upper_bound=5, seed = 200, budget_factor_per_dim = 10)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert result[colnames].equals(feature_values[colnames])

@pytest.mark.skip(reason='Setting seeds is currently not possible in LS features')
def test_calculate_length_scales_features(feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            f = get_problem(fid, 1, dim)
            features = calculate_length_scales_features(f, dim, lower_bound=-5, upper_bound=5, seed = 200, budget_factor_per_dim = 10)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert result[colnames].equals(feature_values[colnames])

def test_calculate_sobol_indices_features(feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            f = get_problem(fid, 1, dim)
            features = calculate_sobol_indices_features(f, dim, lower_bound=-5, upper_bound=5, seed = 200, sampling_coefficient = 100)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert result[colnames].equals(feature_values[colnames])
