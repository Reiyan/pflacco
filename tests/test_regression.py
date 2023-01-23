import numpy as np
import pandas as pd


def test_validate_variable_types_column_names():
    # Make sure _validate_variable_types does not change column names of 
    # `X` argument.
    from pflacco.utils import _validate_variable_types
    from pflacco.sampling import create_initial_sample
    
    X = create_initial_sample(2, sample_type = 'lhs')
    X.columns = ["a", "b"]
    y = pd.Series(np.zeros(len(X)), name="z")
    XX, yy = _validate_variable_types(X, y)
     
    # X checks
    assert all(X.columns == ["a", "b"])
    assert all(XX.columns == ["x0", "x1"])

    # Y checks
    assert y.name == "z"
    assert yy.name == "y"
