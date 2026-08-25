import os
import numpy as np
import pandas as pd
import pytest
import platform

from ioh import get_problem
from pandas.testing import assert_frame_equal

from .conftest import _read_fixture
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
    X = _read_fixture(os.path.join(RSC, 'test_classical_ela_features.pkl'))
    return X

@pytest.fixture
def cm_feature_values():
    X = _read_fixture(os.path.join(RSC, 'test_cm_ela_features.pkl'))
    return X

########################################################
# Deterministic Features
def test_calculate_ela_meta(x_samples, feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim)
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
            features = calculate_ela_curvate(tmp, y, f, dim, seed = 100)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
            y = tmp.apply(lambda x: f(x.values), axis = 1)
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
        y = tmp.apply(lambda x: f(x.values), axis = 1)
        calculate_limo(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 3)

def test_block_value_prerequisite_uses_cell_count(x_samples):
    fid = 1
    dim = 2
    tmp = x_samples.iloc[:(dim*50), :dim]
    f = get_problem(fid, 1, dim)
    y = tmp.apply(lambda x: f(x.values), axis = 1)

    # 5 x 5 cells for 100 observations: 4 observations per cell, therefore admissible.
    calculate_limo(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 5)
    # 6 x 6 cells for 100 observations: less than 3 observations per cell.
    with pytest.raises(ValueError, match='The provided value for "block" is too large, resulting in less than 3 observations per cell.'):
        calculate_limo(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 6)
    # An anisotropic grid is judged by its total number of cells (10 x 3 = 30 cells).
    calculate_limo(tmp, y, lower_bound = -5, upper_bound = 5, blocks = [10, 3])

    # A 5-dimensional 4^5 grid needs 3072 observations, not the 3 * 5^5 the old formula implied.
    dim = 5
    tmp = x_samples.iloc[:, :dim]
    f = get_problem(fid, 1, dim)
    y = tmp.apply(lambda x: f(x.values), axis = 1)
    with pytest.raises(ValueError, match = 'too large'):
        calculate_limo(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 4)

def test_block_value_too_low(x_samples):
    with pytest.raises(ValueError, match='The cell convexity features can only be computed when all dimensions have more than 2 cells.'):
        fid = 1
        dim = 2
        tmp = x_samples.iloc[:(dim*50), :dim]
        f = get_problem(fid, 1, dim)
        y = tmp.apply(lambda x: f(x.values), axis = 1)
        calculate_cm_angle(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 2)

def test_block_value_forced(x_samples):
    with pytest.warns(UserWarning, match=r'For the given dataframe X, blocks \[.*\] require at least \d+ observations to retain 3 observations per cell. X only provides \d+.'):
        fid = 1
        dim = 3
        tmp = x_samples.iloc[:(dim*50), :dim]
        f = get_problem(fid, 1, dim)
        y = tmp.apply(lambda x: f(x.values), axis = 1)
        calculate_cm_conv(tmp, y, lower_bound = -5, upper_bound = 5, blocks = 5, force = True)

# https://github.com/Reiyan/pflacco/issues/38
def test_information_content_with_duplicated_samples():
    np.random.seed(0)
    x = np.random.rand(10, 2)
    X = pd.DataFrame(np.vstack([x, x, np.random.rand(10, 2)]), columns = ['x0', 'x1'])

    # Identical decision vectors with differing objective values are mean-aggregated.
    y = pd.Series(np.arange(X.shape[0], dtype = float))
    assert np.isfinite(calculate_information_content(X, y, seed = 100)['ic.h_max'])

    # Complete duplicates (decision vector and objective value) are dropped.
    y = X.apply(lambda r: float(np.sum(r.values ** 2)), axis = 1)
    assert np.isfinite(calculate_information_content(X, y, seed = 100)['ic.h_max'])

# Sphere, dim 2, x in [-5.12, 5.12]: ||grad|| = 2*||x|| <= 2*sqrt(2)*5.12 ~ 14.48
# and the Hessian is 2*I, so its condition number is exactly 1.
def test_ela_curv_matches_analytical_sphere():
    f = lambda x: float(np.sum(np.asarray(x) ** 2))
    X = create_initial_sample(2, 200, lower_bound = -5.12, upper_bound = 5.12, seed = 42)
    y = X.apply(lambda x: f(x.values), axis = 1)
    features = calculate_ela_curvate(X, y, f, 2, seed = 42, normalize = False)

    assert features['ela_curv.grad_norm.max'] <= 14.49
    assert np.isclose(features['ela_curv.hessian_cond.max'], 1)

# Objective normalization, cf. Prager and Trautmann (2023).
@pytest.mark.parametrize('feature_set', ['ela_meta', 'ic', 'limo', 'ela_curv'])
def test_normalize_makes_features_shift_and_scale_invariant(feature_set, x_samples):
    dim = 2
    X = x_samples.iloc[:(dim*50), :dim]
    f = get_problem(1, 1, dim)
    y = X.apply(lambda x: f(x.values), axis = 1)
    g = lambda x: 3.7 * f(x) - 100
    calls = {
        'ela_meta':  lambda fn, yy: calculate_ela_meta(X, yy, normalize = True),
        'ic':        lambda fn, yy: calculate_information_content(X, yy, seed = 1, normalize = True),
        'limo':      lambda fn, yy: calculate_limo(X, yy, lower_bound = -5, upper_bound = 5, blocks = 3, normalize = True),
        'ela_curv':  lambda fn, yy: calculate_ela_curvate(X, yy, fn, dim, seed = 1, normalize = True),
    }
    original = calls[feature_set](f, y)
    shifted = calls[feature_set](g, 3.7 * y - 100)

    for key in original:
        if 'costs' in key:
            continue
        assert np.isclose(original[key], shifted[key], rtol = 1e-6, equal_nan = True), key
