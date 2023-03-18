import os
import pandas as pd
import pytest
from ioh import get_problem
from pflacco.local_optima_network_features import compute_local_optima_network, calculate_lon_features

DIMS = [2, 5, 10]
RSC = os.path.join('tests', 'resources')

@pytest.fixture
def feature_values():
    X = pd.read_pickle(os.path.join(RSC, 'test_lon_features.pkl'))
    return X

@pytest.mark.skip(reason='To be implemented')
def test_calculate_lon_features(feature_values):
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            f = get_problem(fid, 1, dim)
            nodes, edges = compute_local_optima_network(f, dim, lower_bound=-5, upper_bound=5, seed = 200, basin_hopping_iteration = 10, stopping_threshold= 100)
            features = calculate_lon_features(nodes, edges)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            result.append(data)
    result = pd.concat(result).reset_index(drop = True)
    colnames = result.columns[~result.columns.str.contains('costs_runtime')]
    assert result[colnames].equals(feature_values[colnames])
