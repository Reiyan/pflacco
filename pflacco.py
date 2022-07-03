import numpy as np
import os
import pandas as pd
import time

from datetime import timedelta
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from scipy.stats import entropy, moment
from SALib.analyze import sobol

from bbob import bbobbenchmarks

from pflacco_utils import _transform_bounds_to_canonical, _validate_variable_types, _determine_max_n_blocks, _check_blocks_variable, _create_blocks
from sampling import _create_local_search_sample, create_initial_sample, _levy_random_walk
from classical_ela_features import *
from local_optima_network_features import *

allowed_ela_features = ['limo', 'ela_meta', 'ela_level', 'ela_distr', 'disp', 'ic', 'pca', 'nbc', 'fd']


def calculate_hill_climbing_features(f, dim, lower_bound, upper_bound, n_runs = 100, budget_factor_per_run = 1000, method = 'L-BFGS-B', minimize = True, seed = None, minkowski_p = 2):
      start_time = time.monotonic()
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

      opt_result, nfvals = _create_local_search_sample(f, dim, lower_bound, upper_bound, n_runs = n_runs, budget_factor_per_run=budget_factor_per_run, method = method, minimize = minimize, seed = seed)

      cdist_mat = pdist(opt_result, metric='minkowski', p = minkowski_p)
      dist_mean = cdist_mat.mean()
      dist_std = cdist_mat.std(ddof = 1)

      dist_mat = squareform(cdist_mat)
      best_optimum_idx = opt_result[:, dim] == opt_result[: , dim].min()
      dist_global_local_mean = dist_mat[best_optimum_idx, :].mean()
      dist_global_local_std = dist_mat[best_optimum_idx, :].std(ddof = 1)

      return {
            'hill_climbing.avg_dist_between_opt': dist_mean,
            'hill_climbing.std_dist_between_opt': dist_std,
            'hill_climbing.avg_dist_local_to_global': dist_global_local_mean,
            'hill_climbing.std_dist_local_to_global': dist_global_local_std,
            'hill_climbing.additional_function_eval': nfvals,
            'hill_climbing.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }
      
def calculate_gradient_features(f, dim, lower_bound, upper_bound, step_size = None, budget_per_random_walk = 1000, seed = None):
      start_time = time.monotonic()
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

      if seed is not None:
            np.random.seed(seed)

      if step_size is None:
            step_size = np.array([(upper_bound[i] - lower_bound[i]) * dim /1000 for i in range(dim)])
      elif not isinstance(step_size, list) and not isinstance(step_size, np.array):
            step_size = np.array([step_size] * dim)

      nfev = 0
      g_avgs = []
      g_devs = []
      for _ in range(dim):
            result = []
            dd = np.random.choice([0, 1], size = dim)
            bounds = list(zip(lower_bound, upper_bound))
            x = np.array([bounds[x][dd[x]] for x in range(dim)], dtype = 'float64')
            nfev += 1
            fval = f(x)
            signs = np.array([1 if x == 0 else -1 for x in dd])
            result.append(np.append(x, fval))
            for i in range(budget_per_random_walk - 1):
                  cd = np.random.choice(range(dim))
                  if not (x[cd] + signs[cd]*step_size[cd] <= bounds[cd][1] and x[cd] + signs[cd]*step_size[cd] >= bounds[cd][0]):
                        signs[cd] = signs[cd] * -1
                  x[cd] = x[cd] + signs[cd]*step_size[cd]
                  fval = f(x)
                  nfev += 1
                  result.append(np.append(x, fval))
      
            result = np.array(result)
            fvals = result[: , dim]
            norm_fval = fvals.max() - fvals.min()
            sp_range = sum([x[1] - x[0] for x in bounds])
            denom = step_size.mean()/sp_range

            g_t = []
            for i in range(len(result) - 1):
                  numer = (fvals[i + 1] - fvals[i]) / norm_fval
                  g_t.append(numer/denom)

            g_t = np.array(g_t)
            g_avg = np.abs(g_t).sum()/len(g_t)
            g_avgs.append(g_avg)

            g_dev_num = sum([(g_avg - np.abs(g))**2 for g in g_t])
            g_dev = np.sqrt(g_dev_num/(len(g_t) - 1))
            g_devs.append(g_dev)
      
      g_avgs = np.array(g_avgs)
      g_devs = np.array(g_devs)

      return {
            'gradient.g_avg': g_avgs.mean(),
            'gradient.g_std': g_devs.mean(),
            'gradient.additional_function_eval': nfev,
            'gradient.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_fitness_distance_correlation(X, y, f_opt = None, proportion_of_best = 1, minimize = True, minkowski_p = 2):
      start_time = time.monotonic()
      if proportion_of_best > 1 or proportion_of_best <= 0:
            raise Exception('Proportion of the best samples must be in the interval (0, 1]')
      if not type(y) is not pd.Series:
            y = pd.Series(y)
      else:
            y = y.reset_index(drop = True)
      if not minimize:
            y = y * -1
      if f_opt is not None and not minimize:
            f_opt = -f_opt
      if f_opt is None:
            fopt_idx = y.idxmin()
      elif len(y[y == f_opt]) > 0:
            fopt_idx = y[y == f_opt].index[0]
      else:
            fopt_idx = y.idxmin()

      if proportion_of_best < 1:
            sorted_idx = y.sort_values().index
            if round(len(sorted_idx)*proportion_of_best) < 2:
                  raise Exception(f'Selecting only {proportion_of_best} of the sample results in less than 2 remaining observations.')
            sorted_idx = sorted_idx[:round(len(sorted_idx)*proportion_of_best)]
            X = X.iloc[sorted_idx].reset_index(drop = True)
            y = y[sorted_idx].reset_index(drop = True)

      dist = squareform(pdist(X, metric = 'minkowski', p = minkowski_p))[fopt_idx]
      dist_mean = dist.mean()
      y_mean = y.mean()

      cfd = np.array([(y[i] - y_mean)*(dist[i] - dist_mean) for i in range(len(y))])
      cfd = cfd.sum()/len(y)

      rfd = cfd/(y.std(ddof = 1) * np.std(dist, ddof = 1))

      return {
            'fitness_distance.fd_correlation': rfd,
            'fitness_distance.fd_cov': cfd,
            'fitness_distance.distance_mean': dist_mean,
            'fitness_distance.distance_std': np.std(dist, ddof = 1),
            'fitness_distance.fitness_mean': y_mean,
            'fitness_distance.fitness_std': y.std(ddof = 1),
            'fitness_distance.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }
         
def calculate_length_scales_features(f, dim, lower_bound, upper_bound, budget_factor_per_dim = 1000, seed = None, minimize = True, sample_size_from_kde = 500):
      start_time = time.monotonic()
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

      if seed is not None:
            np.random.seed(seed)

      bounds = list(zip(lower_bound, upper_bound))

      x = np.random.uniform(lower_bound, upper_bound, dim)
      result = []
      nfev = 0
      for _ in range(budget_factor_per_dim * (dim ** 2)):
            x = _levy_random_walk(x)
            x = np.array([np.clip(x[i], bounds[i][0], bounds[i][1]) for i in range(len(x))])
            fval = f(x)
            nfev += 1
            result.append(np.append(x, fval))
      result = np.array(result)
      r_dist = pdist(result[:, :dim])
      r_fval = pdist(result[:, dim].reshape(len(result), 1), metric = 'cityblock')
      r = r_fval/r_dist
      r = r[~np.isnan(r)]
      kernel = gaussian_kde(r)
      sample = np.random.uniform(low=r.min(), high=r.max(), size = sample_size_from_kde)
      prob = kernel.pdf(sample)
      h_r = entropy(prob, base = 2)
      #moment_sample = kernel.resample(sample_size_from_kde*dim).reshape(sample_size_from_kde*dim)
      moments = moment(r, moment = [2, 3, 4])
      return {
            'length_scale.shanon_entropy': h_r,
            'length_scale.mean': np.mean(r),
            'length_scale.std': np.std(r, ddof=1), 
            'length_scale.distribution.second_moment': moments[0],
            'length_scale.distribution.third_moment': moments[1],
            'length_scale.distribution.fourth_moment': moments[2],
            'length_scale.additional_function_eval': nfev,
            'length_scale.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_sobol_indices_features(f, dim, lower_bound, upper_bound, sampling_coefficient = 10000, n_bins = 20, min_obs_per_bin_factor = 1.5, seed = None):
      start_time = time.monotonic()
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)
      if seed is not None:
            np.random.seed(seed)

      X = create_initial_sample(dim, n = sampling_coefficient * (dim + 2), lower_bound = lower_bound, upper_bound = upper_bound, sample_type = 'sobol')
      y = X.apply(lambda x: f(x.values), axis = 1).values

      ## A. Metrics based on Sobol Indices
      pdef = {
            'num_vars': dim,
            'names': X.columns,
            'bounds': list(zip(lower_bound, upper_bound))
      }

      sens = sobol.analyze(pdef, y, print_to_console=False, calc_second_order=False)
      v_inter = 1 - sens['S1'].sum()
      v_cv = sens['ST'].std()/sens['ST'].mean()

      ## B. Fitness- and State-Variance
      # 1. Fitness Variance
      y_hat = y/y.mean()
      mu_2 = y_hat.var()

      # 2. State Variance
      full_sample = X.copy()
      full_sample['y'] = y
      full_sample = full_sample.sort_values('y')
      full_sample['bins'] = pd.cut(full_sample.y, n_bins, labels= range(n_bins))

      d_b_set = []
      d_b_j_set = []
      obs_per_bin = []
      for bin in range(n_bins):
            group = full_sample[full_sample.bins == bin]
            obs_per_bin.append(group.shape[0])
            if group.shape[0] < min_obs_per_bin_factor * dim:
                  d_b_set.append(0)
            else:
                  grp_x = group.to_numpy()[:, :dim]
                  x_mean = grp_x.mean(axis = 0)
                  d_b_j = np.sqrt((grp_x - x_mean) ** 2).mean(axis = 1)
                  d_b_j_set.append(d_b_j)
                  d_b_set.append(d_b_j.mean())
            
      d_b_set = np.array(d_b_set)
      d_b_j_set = np.hstack(np.array(d_b_j_set))
      obs_per_bin = np.array(obs_per_bin)

      d_distribution = np.hstack(np.array([np.array([d_b_set[i]] * obs_per_bin[i]) for i in range(n_bins)]))
      u_2_d = d_distribution.var()

      ## C. Fitness- and State Skewness
      # 1. Fitness Skewness
      norm_factor = np.abs((y.max() - y.min())/2)
      y_hat = ((y.max()- y.min())/2) + y.min()
      fit_skewness =  np.array([(y_hat - y_i)/norm_factor for y_i in y]).mean()

      # 2. State Skewness
      #d_caron = 0.5 - 0.5/n_bins
      norm_factor = np.abs((d_b_j_set.max() - d_b_j_set.min())/2)
      d_caron = (d_b_j_set.max() - d_b_j_set.min())/2 + d_b_j_set.min()
      s_d = np.array([(d_caron - x)/norm_factor for x in d_b_j_set]).mean()

      return {
            'fla_metrics.sobol_indices.degree_of_variable_interaction': v_inter,
            'fla_metrics.sobol_indices.coeff_var_x_sensitivy': v_cv,
            'fla_metrics.fitness_variance': mu_2,
            'fla_metrics.state_variance': u_2_d,
            'fla_metrics.fitness_skewness': fit_skewness,
            'fla_metrics.state_skewness': s_d, 
            'fla_metrics.additional_function_eval' : sampling_coefficient * (dim + 2),
            'fla_metrics.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_features(X, y, lower_bound, upper_bound, features = [], blocks = None,  minimize = True):
      
      if (not isinstance(features, list) and type(blocks) is not np.ndarray) or len(features) == 0:
            raise Exception('No features specified or not provided within list of strings')
      
      if blocks is None:
            blocks = _determine_max_n_blocks(X)
                  
      result = []
      for feature in features:
            if feature not in allowed_ela_features:
                  raise Exception(f'[{feature}] is not a valid ELA feature.')
            if feature == 'disp':
                  result.append(calculate_dispersion(X, y))
            elif feature == 'ela_meta':
                  result.append(calculate_ela_meta(X, y))
            elif feature == 'ela_distr':
                  result.append(calculate_ela_distribution(X, y))
            elif feature == 'ic':
                  result.append(calculate_information_content(X, y))
            elif feature == 'limo':
                  result.append(calculate_limo(X, y, lower_bound = lower_bound, upper_bound = upper_bound, blocks = blocks))
            elif feature == 'nbc':
                  result.append(calculate_nbc(X, y, minimize = minimize))
            elif feature == 'ela_level':
                  result.append(calculate_ela_level(X, y))
            else:
                  result.append(calculate_pca(X, y))

      return dict((key, dict_[key]) for dict_ in result for key in dict_)


'''
test = pd.read_csv('pflacco_5D.csv')
#X = test[['x1', 'x2']]
X = test[['x1', 'x2', 'x3', 'x4', 'x5']]
y = test['y']

def test_obj(X):

      return sum(X) ** 2/(np.sqrt(np.abs(np.sum(X))) + 1)

'''

f = bbobbenchmarks.instantiate(3, iinstance=1)[0]
X = create_initial_sample(5, n = 500, lower_bound = -5, upper_bound = 5, sample_type = 'lhs')
y = X.apply(lambda x: f(x.values), axis = 1)
#result = calculate_sobol_indices_features(test_obj, 10, -5, 5, seed = 100)
grad = calculate_gradient_features(f, 5, -5, 5, seed = 100)

hc = calculate_hill_climbing_features(f, 5, -5, 5, seed = 100)
fdc = calculate_fitness_distance_correlation(X, y)
si = calculate_sobol_indices_features(f, 5, -5, 5, 100)
#ls = calculate_length_scales_features(f, 10, -5, 5, minimize=True, seed=100)

meta = calculate_ela_meta(X, y)
pca = calculate_pca(X, y)
nbc = calculate_nbc(X, y)
disp = calculate_dispersion(X, y)
ic = calculate_information_content(X, y)
distr = calculate_ela_distribution(X, y)
limo = calculate_limo(X, y, -5, 5, blocks = 3)
cm_angle = calculate_cm_angle(X, y, -5, 5, blocks = 3)
cm_conv = calculate_cm_conv(X, y, -5, 5, blocks = 3)
cm_grad = calculate_cm_grad(X, y, -5, 5, blocks = 3)
ela_conv = calculate_ela_conv(X, y, f)
ela_level = calculate_ela_level(X, y)
ela_curvate = calculate_ela_curvate(X, y, f, 5, -5, 5)
ela_local = calculate_ela_local(X, y, f, 5, -5, 5)

nodes, edges = compute_local_optima_network(f, 5, -5, 5)
lon = compute_lon_features(nodes, edges)
print('test')
'''
import os
files = os.listdir('./bbob_test_samples')

for file_ in files:
      sample = pd.read_csv('./bbob_test_samples/' + file_)
      X = sample[['x1', 'x2', 'x3', 'x4', 'x5']]
      y = sample['y']           
      result = calculate_features(X, y, features = ['disp', 'ela_meta', 'ela_distr', 'ic', 'limo', 'nbc'])
      print(f'Successful computation of {file_}')

'''

