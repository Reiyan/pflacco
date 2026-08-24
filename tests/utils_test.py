import numpy as np
import pandas as pd
import pytest

from pflacco.utils import _normalize_objective, _normalize_objective_with_scale

def test_normalize_objective():
    y = pd.Series([-3.0, 1.0, 5.0])
    assert np.allclose(_normalize_objective(y), [0.0, 0.5, 1.0])

def test_normalize_objective_with_scale():
    y = pd.Series([-3.0, 1.0, 5.0])
    y_norm, y_min, y_range = _normalize_objective_with_scale(y)

    assert np.allclose(y_norm, [0.0, 0.5, 1.0])
    assert (y_min, y_range) == (-3.0, 8.0)
    # The offset and range have to reproduce the very same transformation for `f`.
    assert np.allclose((y - y_min)/y_range, y_norm)

def test_normalize_constant_objective_yields_nan():
    y = pd.Series([42.0] * 5)
    with pytest.warns(UserWarning, match = 'constant'):
        y_norm, y_min, y_range = _normalize_objective_with_scale(y)

    assert y_norm.isna().all()
    assert y_min == 42.0 and np.isnan(y_range)
