import numpy as np
import pandas as pd
import time

from datetime import timedelta
from SALib.analyze import sobol
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, gaussian_kde, moment
from typing import Callable, Dict, List, Optional, Union

from .utils import _transform_bounds_to_canonical, _validate_variable_types, _determine_max_n_blocks, _check_blocks_variable, _create_blocks
from .sampling import _create_local_search_sample, create_initial_sample, _levy_random_walk

def calculate_hill_climbing_features(
      f: Callable[[List[float]], float],
      dim: int,
      lower_bound: Union[List[float], float],
      upper_bound: Union[List[float], float],
      n_runs: int = 100,
      budget_factor_per_run: int = 1000,
      method: str = 'L-BFGS-B',
      minimize: bool = True,
      seed: Optional[int] = None,
      minkowski_p: int = 2) -> Dict[str, Union[int, float]]:
      """Calculation of a Hill Climbing features in accordance to [1].

      Parameters
      ----------
      f : Callable[[List[float]], float]
          Objective function to be optimized.
      dim : int
          Dimensionality of the decision space.
      lower_bound : Union[List[float], float]
          Lower bound of variables of the decision space.
      upper_bound : Union[List[float], float]
          Upper bound of variables of the decision space.
      n_runs : int, optional
          Number of independent solver runs to create the sample, by default 100.
      budget_factor_per_run : int, optional
          Budget factor for each individual solver run. The realized budget
          is calculated with ``budget_factor_per_run * dim``, by default 1000.
      method : str, optional
          Type of solver. Any of `scipy.optimize.minimize` can be used, by default 'L-BFGS-B'.
      minimize : bool, optional
          Indicator whether the objective function should be minimized or maximized, by default True.
      seed : Optional[int], optional
          Seed for reproducability, by default None.
      minkowski_p : int, optional
          The p-norm to apply for Minkowski, by default 2.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      References
      ----------
      [1] Abell, T., Malitsky, Y. and Tierney, K., 2013, January.
          Features for exploiting black-box optimization problem structure.
          In International Conference on Learning and Intelligent Optimization (pp. 30-36). 
      """      
      start_time = time.monotonic()
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

      opt_result, nfvals = _create_local_search_sample(f, dim, lower_bound, upper_bound, n_runs = n_runs, budget_factor_per_run=budget_factor_per_run, method = method, minimize = minimize, seed = seed)

      cdist_mat = pdist(opt_result[:, :dim], metric='minkowski', p = minkowski_p)
      dist_mean = cdist_mat.mean()
      dist_std = cdist_mat.std(ddof = 1)

      dist_mat = squareform(cdist_mat)
      best_optimum_idx = opt_result[:, dim] == opt_result[: , dim].min()
      
      # In case of multiple global optima, the distance to the nearest global optima is taken.
      tie_breaker_dist_mat = np.array([dist_mat[best_optimum_idx, x].min() for x in range(dist_mat.shape[1])])
      dist_global_local_mean = tie_breaker_dist_mat.mean()
      dist_global_local_std = tie_breaker_dist_mat.std(ddof = 1)

      return {
            'hill_climbing.avg_dist_between_opt': dist_mean,
            'hill_climbing.std_dist_between_opt': dist_std,
            'hill_climbing.avg_dist_local_to_global': dist_global_local_mean,
            'hill_climbing.std_dist_local_to_global': dist_global_local_std,
            'hill_climbing.additional_function_eval': nfvals,
            'hill_climbing.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }
      
def calculate_gradient_features(
      f: Callable[[List[float]], float], 
      dim: int, 
      lower_bound: Union[List[float], float], 
      upper_bound: Union[List[float], float], 
      step_size: float = None, 
      budget_factor_per_dim: int = 100,
      seed: Optional[int] = None) -> Dict[str, Union[int, float]]:
      """Calculation of a Gradient features in accordance to [1].

      Parameters
      ----------
      f : Callable[[List[float]], float]
          Objective function to be optimized.
      dim : int
          Dimensionality of the decision space.
      lower_bound : Union[List[float], float]
          Lower bound of variables of the decision space.
      upper_bound : Union[List[float], float]
          Upper bound of variables of the decision space.
      step_size : float, optional
          _description_, by default None
      budget_factor_per_dim : int, optional
          The realized budget is calculated with 
          ``budget_factor_per_dim * dim``, by default 100.
      seed : Optional[int], optional
          Seed for reproducability, by default None.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.
      
      References
      ----------
      [1] Malan, K.M. and Engelbrecht, A.P., 2013, June.
          Ruggedness, funnels and gradients in fitness landscapes and the effect on PSO performance.
          In 2013 IEEE Congress on Evolutionary Computation (pp. 963-970). IEEE.
      """

      start_time = time.monotonic()
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

      if seed is not None:
            np.random.seed(seed)

      if step_size is None:
            step_size = (upper_bound.min() - lower_bound.min())/20

      dd = np.random.choice([0, 1], size = dim)
      bounds = list(zip(lower_bound, upper_bound))
      x = np.array([bounds[x][dd[x]] for x in range(dim)], dtype = 'float64')
      nfev = 1
      fval = f(x)
      signs = np.array([1 if x == 0 else -1 for x in dd])
      result = [np.append(x, fval)]
      for i in range(budget_factor_per_dim * dim - 1):
            cd = np.random.choice(range(dim))
            if not (x[cd] + signs[cd]*step_size <= bounds[cd][1] and x[cd] + signs[cd]*step_size >= bounds[cd][0]):
                  signs[cd] = signs[cd] * -1
            x[cd] = x[cd] + signs[cd]*step_size
            fval = f(x)
            nfev += 1
            result.append(np.append(x, fval))
      
      result = np.array(result)
      fvals = result[: , dim]
      norm_fval = fvals.max() - fvals.min()
      sp_range = sum([x[1] - x[0] for x in bounds])
      denom = step_size/sp_range

      g_t = []
      for i in range(len(result) - 1):
            numer = (fvals[i + 1] - fvals[i]) / norm_fval
            g_t.append(numer/denom)
      g_t = np.array(g_t)
      g_avg = np.abs(g_t).sum()/len(g_t)
      g_dev_num = sum([(g_avg - np.abs(g))**2 for g in g_t])
      g_dev = np.sqrt(g_dev_num/(len(g_t) - 1))

      return {
            'gradient.g_avg': g_avg,
            'gradient.g_std': g_dev,
            'gradient.additional_function_eval': nfev,
            'gradient.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_fitness_distance_correlation(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      f_opt: Optional[float] = None,
      proportion_of_best: float = 0.1,
      minimize: bool = True,
      minkowski_p: int = 2) -> Dict[str, Union[int, float]]:
      """Calculation of Fitness Distance Correlation features in accordance to [1] and [2].

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      f_opt : Optional[float], optional
          Objective value of the global optimum (if known), by default None.
      proportion_of_best : float, optional
          Value which is used to split the provided observations `X` and `y` into
          the top `proportion_of_best * 100`% individuals and the remaining.
          Must be within the interval (0, 1], by default 0.1.
      minimize : bool, optional
          Indicator whether the objective function should be minimized or maximized, by default True.
      minkowski_p : int, optional
          The p-norm to apply for Minkowski, by default 2.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      References
      ----------
      [1] Jones, T. and Forrest, S., 1995, July.
          Fitness distance correlation as a measure of problem difficulty for genetic algorithms.
          In ICGA (Vol. 95, pp. 184-192).
      [2] Müller, C.L. and Sbalzarini, I.F., 2011, April.
          Global characterization of the CEC 2005 fitness landscapes using fitness-distance analysis.
          In European conference on the applications of evolutionary computation (pp. 294-303).

      """      
      start_time = time.monotonic()
      if proportion_of_best > 1 or proportion_of_best <= 0:
            raise ValueError('Proportion of the best samples must be in the interval (0, 1]')
      '''
      if not type(y) is not pd.Series:
            y = pd.Series(y)
      else:
            y = y.reset_index(drop = True)
      '''
      X, y = _validate_variable_types(X, y)

      if not minimize:
            y = y * -1
      if f_opt is not None and not minimize:
            f_opt = -f_opt

      if proportion_of_best < 1:
            sorted_idx = y.sort_values().index
            if round(len(sorted_idx)*proportion_of_best) < 2:
                  raise Exception(f'Selecting only {proportion_of_best} of the sample results in less than 2 remaining observations.')
            sorted_idx = sorted_idx[:round(len(sorted_idx)*proportion_of_best)]
            X = X.iloc[sorted_idx].reset_index(drop = True)
            y = y[sorted_idx].reset_index(drop = True)

            if f_opt is None:
                fopt_idx = y.idxmin()
            elif len(y[y == f_opt]) > 0:
                fopt_idx = y[y == f_opt].index[0]
            else:
                fopt_idx = y.idxmin()

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
         
def calculate_length_scales_features(
      f: Callable[[List[float]], float], 
      dim: int,
      lower_bound: Union[List[float], float],
      upper_bound: Union[List[float], float],
      budget_factor_per_dim: int = 100, 
      seed: Optional[int] = None,
      minimize: bool = True,
      sample_size_from_kde: int = 500) -> Dict[str, Union[int, float]]:
      """Calculation of Length-Scale features in accordance to [1].

      Parameters
      ----------
      f : Callable[[List[float]], float]
          Objective function to be optimized.
      dim : int
          Dimensionality of the decision space.
      lower_bound : Union[List[float], float]
          Lower bound of variables of the decision space.
      upper_bound : Union[List[float], float]
          Upper bound of variables of the decision space.
      budget_factor_per_dim : int, optional
          The realized budget is calculated with 
          ``budget_factor_per_dim * (dim ** 2)``, by default 100
      seed : Optional[int], optional
          Seed for reproducability, by default None
      minimize : bool, optional
          Indicator whether the objective function should be minimized or maximized, by default True
      sample_size_from_kde : int, optional
          Sample size which is sampled from the fitted kde distribution, by default 500.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      References
      ----------
      [1] Morgan, R. and Gallagher, M., 2017.
          Analysing and characterising optimization problems using length scale.
          Soft Computing, 21(7), pp.1735-1752.
      """      
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
      r = np.divide(r_fval, r_dist, where=r_dist != 0)
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

def calculate_sobol_indices_features(
      f: Callable[[List[float]], float],
      dim: int,
      lower_bound: Union[List[float], float],
      upper_bound: Union[List[float], float],
      sampling_coefficient: int = 10000,
      n_bins: int = 20,
      min_obs_per_bin_factor: float = 1.5,
      seed: Optional[int] = None) -> Dict[str, Union[int, float]]:
      """Calculation of Sobol Indices, Fitness- and State-Distribution features.

      Parameters
      ----------
      f : Callable[[List[float]], float]
          Objective function to be optimized.
      dim : int
          Dimensionality of the decision space.
      lower_bound : Union[List[float], float]
          Lower bound of variables of the decision space.
      upper_bound : Union[List[float], float]
          Upper bound of variables of the decision space.
      sampling_coefficient : int, optional
          Factor which determines the sample size. The actual sample size
          used in the paper is ``sampling_coffient * (dim + 2)``, by default 10000.
      n_bins : int, optional
          Number of bins used in the construction of the histogram, by default 20.
      min_obs_per_bin_factor : float, optional
          Bins with less than ``min_obs_per_bin_factoro * dim``
          are ignored in the computation (see Equation 5 of [1]), by default 1.5.
      seed : Optional[int], optional
          Seed for reproducability, by default None.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      References
      ----------
      [1] Waibel, C., Mavromatidis, G. and Zhang, Y.W., 2020, July.
          Fitness Landscape Analysis Metrics based on Sobol Indices and Fitness-and State-Distributions.
          In 2020 IEEE Congress on Evolutionary Computation (CEC) (pp. 1-8).
      """      
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
      d_b_j_set = np.hstack(d_b_j_set)
      obs_per_bin = np.array(obs_per_bin)

      d_distribution = np.hstack([np.array([d_b_set[i]] * obs_per_bin[i]) for i in range(n_bins)])

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
