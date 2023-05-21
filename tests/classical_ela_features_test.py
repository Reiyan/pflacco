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

@pytest.fixture
def cm_feature_values():
    X = pd.read_pickle(os.path.join(RSC, 'test_cm_ela_features.pkl'))
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
def test_calculate_cm_angle(x_samples, cm_feature_values):
    result = []
    for fid in range(1,25):
        for dim in [2, 3, 5]:
            force = False
            if dim == 5:
                force = True
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_cm_angle(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 3, force = force)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], cm_feature_values[colnames]) is None)

def test_calculate_cm_conv(x_samples, cm_feature_values):
    result = []
    for fid in range(1,25):
        for dim in [2, 3, 5]:
            force = False
            if dim == 5:
                force = True
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_cm_conv(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 3, force = force)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], cm_feature_values[colnames]) is None)

def test_calculate_cm_grad(x_samples, cm_feature_values):
    result = []
    for fid in range(1,25):
        for dim in [2, 3, 5]:
            force = False
            if dim == 5:
                force = True
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_cm_grad(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 3, force = force)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], cm_feature_values[colnames]) is None)

def test_calculate_limo(x_samples, cm_feature_values):
    result = []
    for fid in range(1,25):
        for dim in [2, 3, 5]:
            force = False
            if dim == 5:
                force = True
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x), axis = 1)
            features = calculate_limo(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 3, force = force)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert (assert_frame_equal(result[colnames], cm_feature_values[colnames]) is None)

def test_block_value_prerequisite(x_samples):
    with pytest.raises(ValueError, match='The provided value for "block" is too large, resulting in less than 3 observations per cell.'):
        fid = 1
        dim = 10
        tmp = x_samples.iloc[:(dim*50), :dim]
        f = get_problem(fid, 1, dim)
        y = tmp.apply(lambda x: f(x), axis = 1)
        calculate_limo(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 3)

def test_block_value_too_low(x_samples):
    with pytest.raises(ValueError, match='The cell convexity features can only be computed when all dimensions have more than 2 cells.'):
        fid = 1
        dim = 10
        tmp = x_samples.iloc[:(dim*50), :dim]
        f = get_problem(fid, 1, dim)
        y = tmp.apply(lambda x: f(x), axis = 1)
        calculate_cm_angle(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 2)

def test_block_value_forced(x_samples):
    with pytest.warns(UserWarning, match=r'For the given dataframe X, the recommended maximum number of blocks per dim is \d+. The current value for blocks \(\d+\) exceeds that.'):
        fid = 1
        dim = 3
        tmp = x_samples.iloc[:(dim*50), :dim]
        f = get_problem(fid, 1, dim)
        y = tmp.apply(lambda x: f(x), axis = 1)
        calculate_cm_conv(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 5, force = True)
