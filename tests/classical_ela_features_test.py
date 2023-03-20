import os
import pandas as pd
import pytest
import platform

from ioh import get_problem
from pandas.testing import assert_frame_equal
from pflacco.classical_ela_features import *
from pflacco.sampling import create_initial_sample

DIMS = [2, 5, 10]
RSC = os.path.join('tests', 'resources')
if platform.system() == 'Windows': 
    RSC = os.path.join(RSC, 'windows')
elif platform.system() == 'Linux':
    RSC = os.path.join(RSC, 'linux')
elif platform.system() == 'Darwin':
    RSC = os.path.join(RSC, 'darwin')
else:
    raise RuntimeError('Could not determine the system platform and therefore not load the appropriate test files.')

@pytest.fixture
def feature_values():
    X = pd.read_pickle(os.path.join(RSC, 'test_classical_ela_features.pkl'))
    return X

########################################################
# Deterministic Features
def test_calculate_ela_meta(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_ela_meta(tmp, y)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], feature_values[colnames]) is None)

def test_calculate_ela_distr(x_samples, feature_values):
    dim = x_samples.shape[1] - 1
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_ela_distribution(tmp, y)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], feature_values[colnames]) is None)

def test_calculate_ela_level(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_ela_level(tmp, y)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], feature_values[colnames]) is None)

def test_calculate_nbc(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_nbc(tmp, y)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], feature_values[colnames]) is None)

def test_calculate_disp(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_dispersion(tmp, y)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], feature_values[colnames]) is None)

def test_calculate_pca(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_pca(tmp, y)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], feature_values[colnames]) is None)

########################################################
# Stochastic Features
def test_calculate_ela_local(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_ela_local(tmp, y, f, dim, -5, 5, seed = 100)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], feature_values[colnames]) is None)

def test_calculate_ela_curvate(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_ela_curvate(tmp, y, f, dim, -5, 5, seed = 100)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], feature_values[colnames]) is None)

def test_calculate_ela_conv(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_ela_conv(tmp, y, f, seed = 100)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], feature_values[colnames]) is None)

def test_calculate_information_content(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_information_content(tmp, y, seed = 100)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], feature_values[colnames]) is None)

########################################################
## Cell Mapping Features
@pytest.mark.skip(reason='To be implemented')
def test_calculate_cell_mapping(feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            X = create_initial_sample(dim, n = 3 ** dim)
            f = get_problem(fid, 1, dim)
            y = X.apply(lambda x: f(x), axis = 1)
            features = calculate_information_content(X, y, seed = 100)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert result[colnames].equals(feature_values[colnames])
