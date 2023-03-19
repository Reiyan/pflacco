from pflacco.classical_ela_features import *
from pflacco.misc_features import *
from pflacco.sampling import * 
from pflacco.local_optima_network_features import *
import os
import pandas as pd
from ioh import get_problem, ProblemType
import random
import sys
from pathlib import Path
#from decimal import Decimal
#sys.path[0] = str(Path(sys.path[0]).parent)


DIMS = [2, 5, 10]
RSC = os.path.join('tests', 'resources')

x_samples = pd.read_pickle(os.path.join(RSC, 'init_sample.pkl'))
#x_samples = pd.DataFrame({col: [Decimal(val) for val in x_samples[col]] for col in x_samples.columns})

#ela_deterministic = pd.read_csv(os.path.join(RSC, 'ela_deterministic.csv'))

def gen_sample():
    for dim in DIMS:
        sample = create_initial_sample(dim, lower_bound = -5, upper_bound = 5, seed = 50)
        sample.to_pickle(os.path.join(RSC, f'regular_sample_d{dim}.pkl'))

    sample = create_initial_sample(5, lower_bound = [-1, 3, 5, 2, 1], upper_bound = 10, seed = 50)
    sample.to_pickle(os.path.join(RSC, 'bound_sample.pkl'))

    sample = create_initial_sample(2, sample_type = 'sobol', seed = 50)
    sample.to_pickle(os.path.join(RSC, 'sobol_sample.pkl'))

    sample = create_initial_sample(5, sample_type = 'random', seed = 50)
    sample.to_pickle(os.path.join(RSC, 'random_sample.pkl'))


def gen_classical_features():
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim, ProblemType.BBOB)
            y = tmp.apply(lambda x: f(x), axis = 1)
            ela_meta = calculate_ela_meta(tmp, y)
            ela_distr = calculate_ela_distribution(tmp, y)
            ela_level = calculate_ela_level(tmp, y)
            nbc = calculate_nbc(tmp, y)
            disp = calculate_dispersion(tmp, y)
            pca = calculate_pca(tmp, y)
            ela_local = calculate_ela_local(tmp, y, f, dim, -5, 5, seed = 100)
            ela_curv = calculate_ela_curvate(tmp, y, f, dim, -5, 5, seed = 100)
            ela_conv = calculate_ela_conv(tmp, y, f, seed = 100)
            ic = calculate_information_content(tmp, y, seed = 100)

            data = pd.DataFrame({**ela_local, **ela_curv, **ela_conv, **ic, **ela_meta, **ela_level, **ela_distr, **nbc, **pca, **disp, **{'fid':fid}, **{'dim':dim}}, index = [0])
            result.append(data)
            
    result = pd.concat(result).reset_index(drop=True)
    result = result[result.columns[~result.columns.str.contains('costs_runtime')]]
    result = result.sort_values(by = ['fid', 'dim']).reset_index(drop = True)
    result.to_pickle(os.path.join(RSC, 'test_classical_ela_features.pkl'))

def gen_cell_features():
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            #TODO remove
            dim = 2
            #X = create_initial_sample(dim, n = (dim ** 3) * 3, seed = 100)
            X = x_samples
            f = get_problem(fid, 1, dim, ProblemType.BBOB)
            y = X.apply(lambda x: f(x), axis = 1)
            cm_angle = calculate_cm_angle(X, y, lower_bound = -5, upper_bound = 5, blocks = 3)
            cm_conv = calculate_cm_conv(X, y, lower_bound = -5, upper_bound = 5, blocks = 3)
            cm_grad = calculate_cm_grad(X, y, lower_bound = -5, upper_bound = 5, blocks = 3)
            limo = calculate_limo(X, y, lower_bound = -5, upper_bound = 5, blocks = 3)

            data = pd.DataFrame({**cm_angle, **cm_conv, **cm_grad, **limo, **{'fid':fid}, **{'dim':dim}}, index = [0])
            result.append(data)
            
    result = pd.concat(result).reset_index(drop=True)
    result = result[result.columns[~result.columns.str.contains('costs_runtime')]]
    result = result.sort_values(by = ['fid', 'dim']).reset_index(drop = True)
    result.to_pickle(os.path.join(RSC, 'test_cm_ela_features.pkl'))


def gen_misc_features():
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            tmp = x_samples.iloc[:(dim*50), :dim]
            f = get_problem(fid, 1, dim, ProblemType.BBOB)
            y = tmp.apply(lambda x: f(x), axis = 1)
            hc = calculate_hill_climbing_features(f, dim, lower_bound = -5, upper_bound = 5, seed = 200)
            fd = calculate_fitness_distance_correlation(tmp, y)
            grad = calculate_gradient_features(f, dim, lower_bound=-5, upper_bound=5, seed = 200, budget_factor_per_dim = 10)
            ls = calculate_length_scales_features(f, dim, lower_bound=-5, upper_bound=5, seed = 200, budget_factor_per_dim = 10)
            si = calculate_sobol_indices_features(f, dim, lower_bound=-5, upper_bound=5, seed = 200, sampling_coefficient = 100)
            
            data = pd.DataFrame({**hc, **fd, **grad, **ls, **si, **{'fid':fid}, **{'dim':dim}}, index = [0])
            print(str(fid) + ' - ' + str(dim))
            result.append(data)
            
    result = pd.concat(result).reset_index(drop=True)
    result = result[result.columns[~result.columns.str.contains('costs_runtime')]]
    result = result.sort_values(by = ['fid', 'dim']).reset_index(drop = True)
    result.to_pickle(os.path.join(RSC, 'test_misc_ela_features.pkl'))

def gen_lon_features():
    result = []
    for fid in range(1,25):
        for dim in DIMS:
            f = get_problem(fid, 1, dim, ProblemType.BBOB)
            nodes, edges = compute_local_optima_network(f, dim, lower_bound=-5, upper_bound=5, seed = 200, basin_hopping_iteration = 10, stopping_threshold= 100)
            features = calculate_lon_features(nodes, edges)
            data = pd.DataFrame(features, index = [0])
            data['fid'] = fid
            data['dim'] = dim
            print(str(fid) + ' - ' + str(dim))
            result.append(data)
    result = pd.concat(result).reset_index(drop=True)
    result = result[result.columns[~result.columns.str.contains('costs_runtime')]]
    result = result.sort_values(by = ['fid', 'dim']).reset_index(drop = True)
    result.to_pickle(os.path.join(RSC, 'test_lon_features.pkl'))

def lon_investigation():
    result = []
    for rep in range(10):
        fid = 3
        dim = 2
        f = get_problem(fid, 1, dim, ProblemType.BBOB)
        nodes, edges = compute_local_optima_network(f, dim, lower_bound=-5, upper_bound=5, seed = 200, basin_hopping_iteration = 10, stopping_threshold= 100)
        features = calculate_lon_features(nodes, edges)
        data = pd.DataFrame(features, index = [0])
        data['fid'] = fid
        data['dim'] = dim
        print(str(fid) + ' - ' + str(dim))
        result.append(data)
    result = pd.concat(result).reset_index(drop=True)
    result = result[result.columns[~result.columns.str.contains('costs_runtime')]]
    result = result.sort_values(by = ['fid', 'dim']).reset_index(drop = True)
    print('test')


def ls_investigation():
    result = []
    for rep in range(10):
        dim = 3
        fid = 2
        tmp = x_samples.iloc[:(dim*50), :dim]
        f = get_problem(fid, 1, dim, ProblemType.BBOB)
        y = tmp.apply(lambda x: f(x), axis = 1)
        ls = calculate_length_scales_features(f, dim, lower_bound=-5, upper_bound=5, seed = 200, budget_factor_per_dim = 10)
        data = pd.DataFrame({**ls}, index = [0])
        result.append(data)

    result = pd.concat(result).reset_index(drop=True)
    print('test')

    
            





gen_classical_features()
#gen_sample()
#gen_cell_features()
gen_misc_features()
#ls_investigation()
gen_lon_features()
#lon_investigation()