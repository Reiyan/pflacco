import numpy as np


def test_validate_variable_types_column_names():
    # Make sure _validate_variable_types does not change column names of 
    # `X` argument.
    from pflacco.utils import _validate_variable_types
    from pflacco.sampling import create_initial_sample
    
    X = create_initial_sample(2, sample_type = 'lhs')
    X.columns = ["a", "b"]
    y = np.zeros(len(X))
    XX, yy = _validate_variable_types(X, y)
    assert (X.columns == ["a", "b"]).all()
    assert (XX.columns == ["x0", "x1"]).all()
