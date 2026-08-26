import os
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

RSC = os.path.join('tests', 'resources')

def _read_fixture(path):
    X = pd.read_pickle(path)
    # The pickles store an object-dtype column index, whereas pandas >=3 creates a
    # string-dtype one for freshly built frames. Rebuild it to match the running pandas.
    X.columns = pd.Index(list(X.columns))
    return X

def assert_features_equal(result, expected):
    # Feature values drift in the last digits between library versions, e.g. the
    # condition number of a numerically singular Hessian. assert_frame_equal
    # defaults to rtol=1e-5, which is tighter than that drift; 1e-3 still catches
    # every real regression, the smallest of which changed values several fold.
    assert assert_frame_equal(result, expected, rtol = 1e-3) is None

@pytest.fixture(scope="package")
def x_samples():
    X = pd.read_pickle(os.path.join(RSC, 'init_sample.pkl'))
    return X

@pytest.fixture(scope="package")
def x_mixed_search_space_sample():
    X = [['0.7874226918866236', '0', 'val1'],
         ['0.9734490961401129', '0', 'val1'],
         ['0.0650260710788888', '1', 'val2'],
         ['0.26596579739226944', '0', 'val3'],
         ['0.32845619115593017', '1', 'val1'],
         ['0.9845734020457273', '1', 'val2'],
         ['0.015080181075462606', '1', 'val2'],
         ['0.3737308842941369', '0', 'val3'],
         ['0.23580972536988687', '0', 'val2']]
    X = pd.DataFrame(X, columns = ['x0', 'x1', 'x2'])
    return X