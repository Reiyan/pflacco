import math
import numpy as np
import pandas as pd
import time

from datetime import timedelta
from functools import partial
from numdifftools.core import Gradient, Hessian

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors 
from sklearn.model_selection import StratifiedKFold

from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from scipy.optimize import minimize as scipy_minimize
from scipy.cluster.hierarchy import linkage, cut_tree, _order_cluster_tree

from typing import Callable, Dict, List, Optional, Union

from .utils import _determine_max_n_blocks, _create_blocks, _validate_variable_types, _transform_bounds_to_canonical, _check_blocks_variable, _cartesian_product_efficient


def _calculate_num_derivate(f, lower_bound, upper_bound, delta, eps, zero_tol, r, v, x):
      h0 = np.abs(delta * x) + eps * (np.abs(x) < zero_tol)
      side = 1 * ((x - lower_bound) <= h0) - 1 * ((upper_bound - x) <= h0)
      side = np.array([np.nan if x == 0 else x for x in side])

      grad = np.abs(Gradient(f, method = 'complex')(x))
      if grad.min() > 0:
            gr_scale = grad.max()/grad.min()
            gr_scale_norm = np.sqrt(np.sum(gr_scale) ** 2)

      else:
            gr_scale = np.nan
            gr_scale_norm = np.nan
      
      hess = Hessian(f, method = 'complex')(x)
      eig = np.abs(np.linalg.eig(hess)[0])
      
      if eig.min() > 0:
            hess_cond = eig.max()/eig.min()
      else:
            hess_cond = np.nan
      
      return np.array([gr_scale_norm, gr_scale, hess_cond])

def calculate_ela_meta(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]]) -> Dict[str, Union[int, float]]:
      """Calculation of ela_meta features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.
      """      
      start_time = time.monotonic()

      X, y = _validate_variable_types(X, y)

      # Create liner model and calculate lm features
      model = linear_model.LinearRegression()
      model.fit(X, y)

      lin_simple_intercept = model.intercept_
      lin_simple_coef_min = model.coef_.min()
      lin_simple_coef_max = model.coef_.max()
      lin_simple_coef_max_by_min = lin_simple_coef_max / lin_simple_coef_min
      lin_simple_adj_r2 = 1 - (1 - model.score(X, y)) * (len(y) - 1) / (len(y) - X.shape[1]-1)
      

      # Create linear model with interaction
      # Create pairwise interactions
      X_interact = X.copy()
      for idx in range(len(X.columns)):
            tmp_idx = idx + 1
            while tmp_idx < len(X.columns):         
                  X_interact[X.columns[idx] + X.columns[tmp_idx]] = X[X.columns[idx]] * X[X.columns[tmp_idx]]
                  tmp_idx += 1

      model = linear_model.LinearRegression()
      model.fit(X_interact, y)
      lin_w_interact_adj_r2 = 1 - (1 - model.score(X_interact, y)) * (len(y) - 1) / (len(y) - X_interact.shape[1] - 1)


      # Create quadratic model and calculate qm features
      model = linear_model.LinearRegression()
      X_squared = pd.concat([X, X.pow(2).add_suffix('^2')], axis = 1)
      model.fit(X_squared, y)
      pred = model.predict(X_squared)

      quad_simple_adj_r2 = 1 - (1 - model.score(X_squared, y)) * (len(y) - 1) / (len(y) - X_squared.shape[1]-1)

      quad_model_con_min = np.absolute(model.coef_[int(X_squared.shape[1]/2):]).min()
      quad_model_con_max = np.absolute(model.coef_[int(X_squared.shape[1]/2):]).max()
      quad_simple_cond = quad_model_con_max/quad_model_con_min

      # Create linear model with interaction
      # Create pairwise interactions
      X_interact = X_squared.copy()
      for idx in range(len(X_squared.columns)):
            tmp_idx = idx + 1
            while tmp_idx < len(X_squared.columns):         
                  X_interact[X_squared.columns[idx] + X_squared.columns[tmp_idx]] = X_squared[X_squared.columns[idx]] * X_squared[X_squared.columns[tmp_idx]]
                  tmp_idx += 1

      model = linear_model.LinearRegression()
      model.fit(X_interact, y)
      quad_w_interact_adj_r2 = 1 - (1 - model.score(X_interact, y)) * (len(y) - 1) / (len(y) - X_interact.shape[1]-1)

      return {
            'ela_meta.lin_simple.adj_r2': lin_simple_adj_r2,
            'ela_meta.lin_simple.intercept': lin_simple_intercept,
            'ela_meta.lin_simple.coef.min': lin_simple_coef_min,
            'ela_meta.lin_simple.coef.max': lin_simple_coef_max,
            'ela_meta.lin_simple.coef.max_by_min': lin_simple_coef_max_by_min,
            'ela_meta.lin_w_interact.adj_r2': lin_w_interact_adj_r2,
            'ela_meta.quad_simple.adj_r2': quad_simple_adj_r2,
            'ela_meta.quad_simple.cond': quad_simple_cond,
            'ela_meta.quad_w_interact.adj_r2': quad_w_interact_adj_r2,
            'ela_meta.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_pca(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      prop_cov_x: float = 0.9,
      prop_cor_x: float = 0.9,
      prop_cov_init: float = 0.9,
      prop_cor_init: float = 0.9) -> Dict[str, Union[int, float]]:
      """Calculation of Principal Component features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      prop_cov_x : float, optional
          Proportion of the explained variance by the first
          PC based on the covariance matrix, by default 0.9.
      prop_cor_x : float, optional
          Proportion of the explained variance by the first
          PC based on the correlation matrix, by default 0.9.
      prop_cov_init : float, optional
          Proportion of the explained variance by the first
          PC based on the covariance matrix, by default 0.9.
      prop_cor_init : float, optional
          Proportion of the explained variance by the first
          PC based on the correlation matrix, by default 0.9.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.
      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)

      df = pd.concat([X,y], axis = 1)
      
      # cov_x
      pca = PCA(n_components=2)
      pca.fit(X)
      exp1_var_pc1_cov_x = pca.explained_variance_ratio_[0]

      ev = np.sort(np.linalg.eig(X.cov())[0])[::-1]
      expl_var_cov_x = np.cumsum(ev) / sum(ev)
      idx_list = np.array([idx for idx, element in enumerate(expl_var_cov_x) if element >= prop_cov_x]) + 1
      expl_var_cov_x = min(idx_list) / X.shape[1]

      # cor_x
      X = (X - X.mean()) / X.std()
      pca = PCA(n_components=2)
      pca.fit(X)
      exp1_var_pc1_cor_x = pca.explained_variance_ratio_[0]

      ev = np.sort(np.linalg.eig(X.cov())[0])[::-1]
      expl_var_cor_x = np.cumsum(ev) / sum(ev)
      idx_list = np.array([idx for idx, element in enumerate(expl_var_cor_x) if element >= prop_cor_x]) + 1
      expl_var_cor_x = min(idx_list) / X.shape[1]

      # cov_init
      pca = PCA(n_components=2)
      pca.fit(df)
      exp1_var_pc1_cov_init = pca.explained_variance_ratio_[0]

      ev = np.sort(np.linalg.eig(df.cov())[0])[::-1]
      expl_var_cov_init = np.cumsum(ev) / sum(ev)
      idx_list = np.array([idx for idx, element in enumerate(expl_var_cov_init) if element >= prop_cov_init]) + 1
      expl_var_cov_init = min(idx_list) / df.shape[1]

      # cor_init
      df = (df - df.mean())/df.std()
      pca = PCA(n_components=2)
      pca.fit(df)
      expl_var_pc1_cor_init = pca.explained_variance_ratio_[0]

      ev = np.sort(np.linalg.eig(df.cov())[0])[::-1]
      expl_var_cor_init = np.cumsum(ev) / sum(ev)
      idx_list = np.array([idx for idx, element in enumerate(expl_var_cor_init) if element >= prop_cor_init]) + 1
      expl_var_cor_init = min(idx_list) / df.shape[1]


      return {
          'pca.expl_var.cov_x': expl_var_cov_x,
          'pca.expl_var.cor_x': expl_var_cor_x,
          'pca.expl_var.cov_init': expl_var_cov_init,
          'pca.expl_var.cor_init': expl_var_cor_init,
          'pca.expl_var_PC1.cov_x': exp1_var_pc1_cov_x,
          'pca.expl_var_PC1.cor_x': exp1_var_pc1_cor_x,
          'pca.expl_var_PC1.cov_init': exp1_var_pc1_cov_init,
          'pca.expl_var_PC1.cor_init': expl_var_pc1_cor_init,
          'pca.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_nbc(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      fast_k: float = 0.05,
      dist_tie_breaker: str = 'sample',
      minimize: bool = True) -> Dict[str, Union[int, float]]:
      """Calculation of Nearest Better Clustering features, similar to the R-package `flacco`.


      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      fast_k : float, optional
          Controls the percentage of observations that should be considered when looking
          for the nearest better neighbour, by default 0.05.
      dist_tie_breaker : str, optional
          Strategy to break ties between observations. Currently allows `sample`, by default 'sample'.
      minimize : bool, optional
          Indicator whether the objective function should be minimized or maximized, by default True.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.
      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)

      if fast_k < 1:
            fast_k = math.ceil(fast_k * X.shape[0])
      if fast_k < 0 or fast_k > X.shape[0]:
            raise ValueError(f'[{fast_k}] of "fast_k" does not lie in the interval [0,n] where n is the number of observations.')
      if minimize == False:
            y = y * -1

      nbrs = NearestNeighbors(n_neighbors = fast_k, algorithm='kd_tree').fit(X)
      distances, indices = nbrs.kneighbors(X)
      results = []
      for idx in range(X.shape[0]):
            y_rec = y[idx]
            ind_nn = indices[idx][1:]
            y_near = y[ind_nn]
            better = y_near < y_rec
            if sum(better) > 0:
                  better = better.reset_index(drop = True)
                  b_idx = better.idxmax()
                  results.append([idx, ind_nn[b_idx], distances[idx][b_idx + 1]])
            else:
                  ind_alt = np.array([x for x in range(X.shape[0]) if x not in ind_nn and x != idx])
                  if sum(y[ind_alt] < y_rec) != 0:
                        ind_alt = ind_alt[y[ind_alt] < y_rec] #TODO: check
                  elif sum(y[ind_alt] == y_rec) != 0:
                        ind_alt = ind_alt[y[ind_alt] == y_rec] #TODO: check
                  else:
                        results.append([idx, None, None])
                        continue
                  if len(ind_alt) == 1:
                        results.append([idx, ind_alt[0], math.sqrt((X.iloc[ind_alt[0]] - X.iloc[idx]).pow(2).sum())])
                  else:
                        d = np.sqrt((X.iloc[ind_alt] - X.iloc[idx]).pow(2).sum(axis = 1)).reset_index(drop = True)
                        if dist_tie_breaker == 'sample':
                              j = np.random.choice(d[d == d.min()].index)
                              results.append([idx, ind_alt[j], d[j]])
                        else:
                              #TODO welche anderen Tiebreaker methoden gibt es? es gibt noch first und last
                              raise ValueError('Currently, the only available tie breaker method is "sample"')

      nb_stats = pd.DataFrame(results, columns = ['ownID', 'nbID', 'nbDist'])
      nb_stats['nearDist'] = [x[1] for x in distances]
      nb_stats['nb_near_ratio'] = nb_stats['nbDist'] / nb_stats['nearDist']
      nb_stats['fitness'] = y
      
      results = []
      for own_id in nb_stats['ownID']:
            x = nb_stats[nb_stats['nbID'] == own_id].index
            count = len(x)
            if count > 0:
                  to_me_dist = nb_stats['nbDist'][x].median()
                  results.append([count, to_me_dist, nb_stats['nbDist'][own_id]/to_me_dist])
            else:
                  results.append([0, None, None])
 
      nb_stats = pd.concat([nb_stats, pd.DataFrame(results, columns = ['toMe_count', 'toMe_dist_median', 'nb_median_toMe_ration'])], axis = 1)
      #nb_stats['nbDist'][nb_stats['nbDist'].isna()] = nb_stats['nearDist'][nb_stats['nbDist'].isna()]
      dist_ratio = nb_stats['nearDist'] / nb_stats['nbDist']

      return {
            'nbc.nn_nb.sd_ratio': nb_stats['nearDist'].std() / nb_stats['nbDist'].std(),
            'nbc.nn_nb.mean_ratio': nb_stats['nearDist'].mean() / nb_stats['nbDist'].mean(),
            'nbc.nn_nb.cor': nb_stats['nearDist'].corr(nb_stats['nbDist']),
            'nbc.dist_ratio.coeff_var': dist_ratio.std()/dist_ratio.mean(),
            'nbc.nb_fitness.cor': nb_stats['toMe_count'].corr(nb_stats['fitness']),
            'nbc.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_dispersion(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      disp_quantiles: List[float] = [0.02, 0.05, 0.1, 0.25],
      dist_method: str = 'euclidean',
      dist_p: int = 2,
      minimize: bool = True) -> Dict[str, Union[int, float]]:
      """Calculation of Dispersion features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      disp_quantiles : List[float], optional
          Quantiles which are used to determine the best elements of the entire sample,
          by default [0.02, 0.05, 0.1, 0.25].
      dist_method : str, optional
          Determines which distance method is used. The given value is passed over to
          `scipy.spatial.distance.pdist`, by default 'euclidean'.
      dist_p : int, optional
          The p-norm to apply for Minkowski. This is only considered when
          `dist_method = 'minkowski'`, by default 2.
      minimize : bool, optional
          Indicator whether the objective function should be minimized or maximized, by default True.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.
      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)

      if minimize == False:
            y = y * -1

      quantiles = np.quantile(y, disp_quantiles)
      
      idx_in_quantiles = {}
      for idx, quantile in enumerate(quantiles):
            idx_in_quantiles[str(disp_quantiles[idx])] = [x for x in y[y < quantile].index]

      dists = {}
      for quantile in idx_in_quantiles:
            idx_in_quantiles[quantile]
            if dist_method == 'minkowski':
                  dists[quantile] = squareform(pdist(X.iloc[idx_in_quantiles[quantile]], metric = dist_method, p = dist_p))
            else:
                  dists[quantile] = squareform(pdist(X.iloc[idx_in_quantiles[quantile]], metric = dist_method))

      dist_full_sample = squareform(pdist(X, metric = dist_method))

      means = []
      medians = []
      for key in dists:
            means.append(np.mean(dists[key][dists[key] != 0]))
            medians.append(np.median(dists[key][dists[key] != 0]))

      keys = ['disp.ratio_mean_{:0>2d}'.format(int(round(x * 100, 0))) for x in disp_quantiles]
      keys.extend(['disp.ratio_median_{:0>2d}'.format(int(round(x * 100, 0))) for x in disp_quantiles])
      keys.extend(['disp.diff_mean_{:0>2d}'.format(int(round(x * 100, 0))) for x in disp_quantiles])
      keys.extend(['disp.diff_median_{:0>2d}'.format(int(round(x * 100, 0))) for x in disp_quantiles])

      values = means / np.mean(dist_full_sample[dist_full_sample != 0])
      values = np.concatenate((values, medians / np.median(dist_full_sample[dist_full_sample != 0])), axis = None)
      values = np.concatenate((values, means - np.mean(dist_full_sample[dist_full_sample != 0])), axis = None)
      values = np.concatenate((values, medians - np.median(dist_full_sample[dist_full_sample != 0])), axis = None)

      result = {keys[i]: values[i] for i in range(len(keys))}
      result['disp.costs_runtime'] = timedelta(seconds=time.monotonic() - start_time).total_seconds()
      return result

def calculate_information_content(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      ic_sorting: str = 'nn',
      ic_nn_neighborhood: int = 20,
      ic_nn_start: Optional[int] = None,
      ic_epsilon: List[float] = np.insert(10 ** np.linspace(start = -5, stop = 15, num = 1000), 0, 0),
      ic_settling_sensitivity: float = 0.05,
      ic_info_sensitivity: float = 0.5,
      seed: Optional[int] = None) -> Dict[str, Union[int, float]]:
      """Calculation of Information Content features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      ic_sorting : str, optional
          Sorting strategy, which is used to define the tour through the landscape.
          Possible values are 'nn' and 'random, by default 'nn'.
      ic_nn_neighborhood : int, optional
          Number of neighbours to be considered in the computation, by default 20.
      ic_nn_start : Optional[int], optional
          Indices of the observation which should be used as starting points.
          When none are supplied, these are chosen randomly, by default None.
      ic_epsilon : List[float], optional
          Epsilon values as described in section V.A of [1],
          by default `np.insert(10 ** np.linspace(start = -5, stop = 15, num = 1000), 0, 0)`.
      ic_settling_sensitivity : float, optional
          Threshold, which should be used for computing the settling sensitivity of [1], by default 0.05.
      ic_info_sensitivity : float, optional
          Portion of partial information sensitivity of [1], by default 0.5
      seed : Optional[int], optional
          Seed for reproducability, by default None

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      References
      ----------
      [1] Muñoz, M.A., Kirley, M. and Halgamuge, S.K., 2014.
          Exploratory landscape analysis of continuous space optimization problems using information content.
          IEEE transactions on evolutionary computation, 19(1), pp.74-87.

      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)

      n = X.shape[1]
      ic_aggregate_duplicated = 'mean'
      if not np.issubdtype(ic_epsilon.dtype, np.number): 
            raise Exception('ic_epsilon contains non numeric data.')
      if ic_epsilon.min() < 0:
            raise Exception('ic_epsilon can only contain numbers in the intervall [0, inf)')
      if 0 not in ic_epsilon:
            raise Exception("One component of ic_epsilon has to be 0. Please add this component")
      if ic_sorting not in ['nn', 'random']:
            raise Exception('Only "nn" and "random" are valid parameter values for "ic_sorting"')
      if ic_settling_sensitivity < 0:
            raise Exception('"ic_settling_sensitivity must be larger than zero')
      if ic_info_sensitivity < -1 or ic_info_sensitivity > 1:
            raise Exception('"ic_settling_sensitivity must be larger than zero')
      if sum(X.duplicated()) == X.shape[0]:
            raise Exception('Can not IC compute information content features, because provided values are identical')
      epsilon = np.unique(ic_epsilon)


      # Duplicate check and mean aggregation for the objective variable, if only variables in the decision space are duplicated.
      dup_index = X.duplicated(keep = False)
      if dup_index.any():
            complete = pd.concat([X, pd.DataFrame(y, columns = ['y'])], axis = 1).duplicated()
            # Remove complete duplicates, because these cannot be aggregated using e.g. the mean of y
            if complete.any():
                  X = X[~complete]
                  y = y[~complete]
                  dup_index = X.duplicated(keep = False)
            
            # TODO Check with Pascal: the next line seems utterly pointless, a flip of the second array is missing. yes double flip.. that is why it is pointless.
            #dup_index = np.bitwise_or(dup_index, np.array(np.flip(dup_index)))
            Z = X[dup_index]
            v = y[dup_index]
            X = X[~dup_index]
            y = y[~dup_index]

            while len(v) > 1:
                  index = np.array([(Z.iloc[0] == Z.iloc[idx]).all() for idx in range(Z.shape[0])])
                  X = pd.concat([X, Z.iloc[[0]]], ignore_index = True)
                  Z = Z[~index]
                  y = pd.concat([y, pd.DataFrame([v[index].mean()])], ignore_index = True)
                  v = v[~index]

            
      if seed is not None and isinstance(seed, int):
            np.random.seed(seed)

      # dist based on ic_sorting
      if ic_sorting == 'random':
            permutation = np.random.choice(range(X.shape[0]), size = X.shape[0], replace = False)
            X = X.iloc[permutation].reset_index(drop = True)
            d = [np.sqrt((X.iloc[idx] - X.iloc[idx + 1]).pow(2).sum()) for idx in range(X.shape[0] - 1)]
      else:
            if ic_nn_start is None:
                  ic_nn_start = np.random.choice(range(X.shape[0]), size = 1)[0]
            if ic_nn_neighborhood < 1 and ic_nn_neighborhood > X.shape[0]:
                  raise Exception(f'[{ic_nn_neighborhood}] is an invalid option for the NN neighborhood, because the sample only covers 1 to {X.shape[0]} observations.')
            nbrs = NearestNeighbors(n_neighbors = min(ic_nn_neighborhood, X.shape[0]), algorithm='kd_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)

            current = ic_nn_start
            candidates = np.delete(np.array([x for x in range(X.shape[0])]), current)
            permutation = [current]
            permutation.extend([None] * (X.shape[0] - 1))
            dists = [None] * (X.shape[0])

            for i in range(1, X.shape[0]):
                  currents = indices[permutation[i-1]]
                  current = np.array([x for x in currents if x in candidates])
                  if len(current) > 0:
                        current = current[0]
                        permutation[i] = current
                        candidates = candidates[candidates != current]
                        dists[i] = distances[permutation[i - 1], currents == current][0]
                  else:
                        nbrs2 = NearestNeighbors(n_neighbors = min(1, X.shape[0])).fit(X.iloc[candidates])
                        distances2, indices2, = nbrs2.kneighbors(X.iloc[permutation[i - 1]].to_numpy().reshape(1, X.shape[1]))
                        current = candidates[np.ravel(indices2)[0]]
                        permutation[i] = current
                        candidates = candidates[candidates != current]
                        dists[i] = np.ravel(distances2)[0]

            d = dists[1:]

      # Calculate psi eps
      psi_eps = []
      y_perm = y[permutation]
      diff_y = np.ediff1d(y_perm)
      ratio = diff_y/d
      for eps in ic_epsilon:
            psi_eps.append([0 if abs(x) < eps else np.sign(x) for x in ratio])

      psi_eps = np.array(psi_eps)
      H = []
      M = []
      for row in psi_eps:
            # Calculate H values
            a = row[:-1]
            b = row[1:]
            probs = []
            probs.append(np.bitwise_and(a == -1, b == 0).mean())
            probs.append(np.bitwise_and(a == -1, b == 1).mean())
            probs.append(np.bitwise_and(a == 0, b == -1).mean())
            probs.append(np.bitwise_and(a == 0, b == 1).mean())
            probs.append(np.bitwise_and(a == 1, b == -1).mean())
            probs.append(np.bitwise_and(a == 1, b == 0).mean())
            H.append(-sum([0 if x == 0 else x * np.log(x)/np.log(6) for x in probs]))

            # Calculate M values
            n = len(row)
            row = row[row != 0]
            len_row = len(row[np.insert(np.ediff1d(row) != 0, 0, False)]) if len(row) > 0 else 0
            M.append(len_row/ (n - 1))

      H = np.array(H)
      M = np.array(M)
      eps_s = epsilon[H < ic_settling_sensitivity]
      eps_s = np.log(eps_s.min()) / np.log(10) if len(eps_s) > 0 else None

      m0 = M[epsilon == 0]
      eps05 = np.where(M > ic_info_sensitivity * m0)[0]
      eps05 = np.log(epsilon[eps05].max()) / np.log(10) if len(eps05) > 0 else None

      return {
            'ic.h_max': H.max(),
            'ic.eps_s': eps_s,
            'ic_eps.max': np.median(epsilon[H == H.max()]),
            'ic.eps_ration': eps05,
            'ic.m0': m0[0],
            'ic.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_ela_distribution(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      #ela_distr_smoothing_bandwith: str = 'SJ',
      #ela_distr_modemass_threshold: float = 0.01,
      ela_distr_skewness_type: int = 3,
      ela_distr_kurtosis_type: int = 3) -> Dict[str, Union[int, float]]:
      """Calculation of ELA Distribution features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      ela_distr_skewness_type : int, optional
          Integer indicating which algorithm to use, by default 3.
      ela_distr_kurtosis_type : int, optional
          Integer indicating which algorithm to use, by default 3.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      """      

      start_time = time.monotonic()
      if ela_distr_skewness_type not in range(1,4):
            raise Exception('Skewness type must be an integer and in the intervall [1,3]')
      if ela_distr_kurtosis_type not in range(1,4):
            raise Exception('Kurtosis type must be an integer and in the intervall [1,3]')
      
      X, y = _validate_variable_types(X, y)
      # Remove NA values
      y = y[~np.isnan(y)]
      n = len(y)

      if n < 4:
            raise Exception('At least 4 complete observations are required')

      # Calculate skewness
      y_skewness = y - y.mean()
      y_skewness = np.sqrt(n) * (y_skewness.pow(3)).sum() / ((y_skewness.pow(2)).sum() ** (3/2))
      if ela_distr_skewness_type == 2:
            y_skewness = y_skewness * np.sqrt(n * (n - 1))/(n - 2)
      elif ela_distr_skewness_type == 3:
            y_skewness = y_skewness * (1 - 1/n) ** (3/2)

      # Calculate kurtosis
      y_kurtosis = y - y.mean()
      r = n * (y_kurtosis.pow(4).sum()) / (y_kurtosis.pow(2).sum() ** 2)
      if ela_distr_kurtosis_type == 1:
            y_kurtosis = r - 3
      elif ela_distr_kurtosis_type == 2:
            y_kurtosis = ((n + 1) * (r - 3) + 6) * (n - 1) / ((n - 2) * (n - 3))
      else:
            y_kurtosis = r * ((1 -1/n) ** 2) - 3

      # Calculate number of peaks
      kernel = gaussian_kde(y)
      low_ = y.min() - 3 * kernel.covariance_factor()
      upp_ = y.max() + 3 * kernel.covariance_factor()
      positions = np.mgrid[low_:upp_:512j]
      d = kernel(positions)

      n = len(d)
      index = np.arange(1, n - 2)
      min_index = np.array([x for x in index if d[x] < d[x - 1] and d[x] < d[x + 1]])
      min_index = np.insert(min_index, 0, 0)
      min_index = np.append(min_index, n)
      
      modemass = []
      for idx in range(len(min_index) - 1):
            a = int(min_index[idx])
            b = int(min_index[idx + 1] - 1)
            modemass.append(d[a:b].mean() + abs(positions[a] - positions[b]))

      n_peaks = (np.array(modemass) > 0.1).sum()

      return {
            'ela_distr.skewness': y_skewness,
            'ela_distr.kurtosis': y_kurtosis,
            'ela_distr.number_of_peaks': n_peaks,
            'ela_distr.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }
            
def calculate_limo(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      lower_bound: Union[List[float], float],
      upper_bound: Union[List[float], float],
      blocks: Optional[Union[List[int], np.ndarray, int]] = None) -> Dict[str, Optional[Union[int, float]]]:
      """Calculation of Linear Model features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      lower_bound : Union[List[float], float]
          Lower bound of variables of the decision space.
      upper_bound : Union[List[float], float]
          Upper bound of variables of the decision space.
      blocks : Optional[Union[List[int], np.ndarray, int]], optional
          Number of blocks per dimension, by default None.

      Returns
      -------
      Dict[str, Optional[Union[int, float]]]
          Dictionary consisting of the calculated features.
      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)
      dims = X.shape[1]
      blocks = _check_blocks_variable(X, dims, blocks)

      # Consolidate X, y, and cells into one data frame
      init = X.copy()
      init['y'] = y
      init['cells'], _ = _create_blocks(X, y, lower_bound, upper_bound, blocks)

      result = {
            'limo.avg_length': None,
            'limo.avg_length_norm': None,
            'limo.length_mean': None,
            'limo.length_sd': None,
            'limo.cor': None,
            'limo.cor_norm': None,
            'limo.ratio_mean': None,
            'limo.ratio_sd': None,
            'limo.sd_ratio_reg': None,
            'limo.sd_ratio_norm': None,
            'limo.sd_mean_reg': None,
            'limo.sd_mean_norm': None
      }

      # if the maximum number of observations in any cell is smaller or equal to the dimensionality, no features can be calculated
      if max([init[init['cells'] == x].shape[0] for x in init['cells'].unique()]) <= dims:
            return result

      coeff_vector = []
      for cell in init['cells'].unique():
            y_cell = init.loc[init['cells'] == cell, 'y']
            X_cell = init.loc[init['cells'] == cell, np.logical_and(init.columns != 'cells', init.columns != 'y')]
            model = linear_model.LinearRegression()
            model.fit(X_cell, y_cell)
            coeff_vector.append(model.coef_)
      coeff_vector = np.array(coeff_vector)
      
      coeff_ratio = np.array([np.max(np.abs(coeff)) / np.min(np.abs(coeff)) if ~np.isnan(coeff).all() else None for coeff in coeff_vector])

      norm_coeff_vector = np.array([coeff / np.sqrt(np.sum(coeff ** 2)) for coeff in coeff_vector])
      length_coeff_vector = np.array([np.sqrt(np.sum(coeff ** 2)) for coeff in coeff_vector])
      sds_unscaled = np.std(coeff_vector, axis = 0, ddof = 1)
      sds_scaled = np.std(norm_coeff_vector, axis = 0, ddof = 1)
      cor_unscaled = np.corrcoef(coeff_vector, rowvar = False)
      cor_scaled = np.corrcoef(norm_coeff_vector, rowvar = False)

      result['limo.avg_length'] = np.sqrt(np.sum(np.mean(coeff_vector, axis=0) ** 2))
      result['limo.avg_length_norm'] = np.sqrt(np.sum(np.mean(norm_coeff_vector, axis=0) ** 2))
      result['limo.length_mean'] = np.mean(length_coeff_vector)
      result['limo.ratio_mean'] = np.mean(coeff_ratio)

      if len(blocks) > 1:
            result['limo.length_sd'] = np.std(length_coeff_vector, ddof = 1)
            result['limo.cor'] = (np.sum(cor_unscaled) - np.sum(np.diag(cor_unscaled))) / (np.ma.count(cor_unscaled) - len(np.diag(cor_unscaled)))
            result['limo.cor_norm'] = (np.sum(cor_scaled) - np.sum(np.diag(cor_scaled))) / (np.ma.count(cor_scaled) - len(np.diag(cor_scaled)))
            result['limo.ratio_sd'] = np.std(coeff_ratio, ddof = 1)
            result['limo.sd_ratio_reg'] = np.max(sds_unscaled) / np.min(sds_unscaled)
            result['limo.sd_ratio_norm'] = np.max(sds_scaled) / np.min(sds_scaled)
            result['limo.sd_mean_reg'] = np.mean(sds_unscaled)
            result['limo.sd_mean_norm'] = np.mean(sds_scaled)
            
      result['limo.costs_runtime'] = timedelta(seconds=time.monotonic() - start_time).total_seconds()

      return result

# TODO small todo inside function
def calculate_cm_angle(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      lower_bound: Union[List[float], float],
      upper_bound: Union[List[float], float],
      blocks: Optional[Union[List[int], np.ndarray, int]] = None,
      minimize: bool = True) -> Dict[str, Union[int, float]]:
      """Calculation of Cell Mapping Angle features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      lower_bound : Union[List[float], float]
          Lower bound of variables of the decision space.
      upper_bound : Union[List[float], float]
          Upper bound of variables of the decision space.
      blocks : Optional[Union[List[int], np.ndarray, int]], optional
          Number of blocks per dimension, by default None.
      minimize : bool, optional
          Indicator whether the objective function should be minimized or maximized, by default True.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)
      dim = X.shape[1]
      if blocks is None:
            blocks = _determine_max_n_blocks(X)
      if not isinstance(blocks, list) and type(blocks) is not np.ndarray:
            blocks = np.array([blocks] * dim)
      elif isinstance(blocks, list):
            blocks = np.array(blocks)
      if len(blocks) != dim:
            raise ValueError('The provided value for "block" does not have the same length as the dimensionality of X.')

      init = X.copy()
      init['y'] = y if minimize == True else -1 * y
      init['cell'], cell_centers = _create_blocks(X, y, lower_bound, upper_bound, blocks)

      grid_best = init.loc[init.groupby('cell')['y'].idxmin()]
      grid_worst = init.loc[init.groupby('cell')['y'].idxmax()]
      y_global_best = grid_best['y'].min()
      y_global_worst = grid_worst['y'].max()
      non_empty = np.sort(grid_best['cell'].unique())
      no_total = np.product(blocks)
      no_empty = no_total - len(non_empty)
      # TODO if no_total = 1
      
      cell_values = []
      for idx in range(len(cell_centers)):
            x_center = cell_centers[idx]
            x_worst = grid_worst.loc[grid_worst['cell'] == idx, X.columns]
            x_best = grid_best.loc[grid_best['cell'] == idx, X.columns]
            if x_worst.shape[0] == 0:
                  cell_values.append([np.nan, np.nan, np.nan, np.nan])
                  continue
            y_local_worst = grid_worst.loc[grid_worst['cell'] == idx, 'y'].values[0]
            y_local_best = grid_best.loc[grid_best['cell'] == idx, 'y'].values[0]
            b2w_ratio = (y_local_worst - y_local_best)/(y_global_worst - y_global_best)
            c2b_vect = (x_best - x_center).to_numpy()
            c2b_dist = np.sqrt(np.sum(c2b_vect ** 2))
            c2w_vect = (x_worst - x_center).to_numpy()
            c2w_dist = np.sqrt(np.sum(c2w_vect ** 2))
            denominator = c2b_dist * c2w_dist
            if denominator == 0 or (x_worst.values == x_best.values).all():
                  angle = 0
            else:
                  x = (c2b_vect * c2w_vect).sum()/denominator
                  angle = np.arccos(x) * 180/np.pi
            cell_values.append([c2b_dist, c2w_dist, angle, b2w_ratio])
      cell_values = np.array(cell_values)
      cv_means = np.nanmean(cell_values, axis = 0)
      cv_sample_stds = np.nanstd(cell_values, axis = 0, ddof = 1)
      return {
            'cm_angle.dist_ctr2best_mean': cv_means[0],
            'cm_angle.dist_ctr2best_sd': cv_sample_stds[0],
            'cm_angle.dist_ctr2worst_mean': cv_means[1],
            'cm_angle.dist_ctr2worst_sd': cv_sample_stds[1],
            'cm_angle.angle_mean': cv_means[2],
            'cm_angle.angle_sd': cv_sample_stds[2],
            'cm_angle.y_ratio_best2worst_mean': cv_means[3],
            'cm_angle.y_ratio_best2worst_sd': cv_sample_stds[3],
            'cm_angle.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_cm_conv(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      lower_bound: Union[List[float], float],
      upper_bound: Union[List[float], float],
      blocks: Optional[Union[List[int], np.ndarray, int]] = None,
      minimize: bool = True,
      cm_conv_diag: bool = False,
      cm_conv_fast_k: float = 0.05) -> Dict[str, Union[int, float]]:
      """Calculation of Cell Mapping Convexity features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      lower_bound : Union[List[float], float]
          Lower bound of variables of the decision space.
      upper_bound : Union[List[float], float]
          Upper bound of variables of the decision space.
      blocks : Optional[Union[List[int], np.ndarray, int]], optional
          Number of blocks per dimension, by default None.
      minimize : bool, optional
          Indicator whether the objective function should be minimized or maximized, by default True.
      cm_conv_diag : bool, optional
          Indicator which, when true, consideres cells on the diagonal also as neighbours, by default False.
      cm_conv_fast_k : float, optional
          Percentage of elements that should be considered within the nearest neighbour computation, by default 0.05.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)
      dim = X.shape[1]
      blocks = _check_blocks_variable(X, dim, blocks)
      if blocks.min() <= 2:
            raise ValueError('The cell convexity features can only be computed when all dimensions have more than 2 cells.')
      if cm_conv_fast_k < 0 or cm_conv_fast_k > X.shape[0]:
            raise ValueError('cm_conv_fast_k must be in the interval [0, n] where n is the number of observations in X.')

      init = X.copy()
      init['y'] = y if minimize == True else -1 * y
      init['cell'], cell_centers = _create_blocks(X, y, lower_bound, upper_bound, blocks)

      # find nearest prototype quick
      if cm_conv_fast_k < 1:
            fast_k = np.ceil(cm_conv_fast_k * X.shape[0])
      else:
            fast_k = cm_conv_fast_k
      fast_k = int(max(2, fast_k))
      n_cells = len(cell_centers)
      nbrs = NearestNeighbors(n_neighbors = fast_k, algorithm='kd_tree').fit(np.vstack([cell_centers, X.to_numpy()]))
      _, indices = nbrs.kneighbors(np.vstack([cell_centers, X.to_numpy()]))
      indices = indices[:n_cells] - n_cells
      nearest_grid_indices = []
      for i in range(n_cells):
            x = np.setdiff1d(indices[i], [i - n_cells], assume_unique=True)
            if len(x[x >= 0]) > 0:
                  x = x[x >= 0][0] 
                  nearest_grid_indices.append(x)
            else:
                  nearest_grid_indices.append(None)
      nearest_grid_indices = np.array(nearest_grid_indices)
      

      # in case none of the nearest observations is a non-cell-center
      all_centers = np.all(indices < 0, axis = 1)
      if all_centers.any():
            n_ctr = all_centers.sum()
            nbrs_backup = NearestNeighbors(n_neighbors = n_ctr + 1, algorithm='kd_tree').fit(np.vstack([cell_centers[all_centers], X.to_numpy()]))
            _, backup_indices = nbrs_backup.kneighbors(np.vstack([cell_centers[all_centers], X.to_numpy()]))
            backup_indices = backup_indices[range(n_ctr), 1:] - n_ctr
            #if len(backup_indices[backup_indices >= 0]) > 0:
            backup_indices = np.array([x[x >= 0][0] for x in backup_indices])
            nearest_grid_indices[all_centers] = backup_indices

      nearest_grid = init.iloc[nearest_grid_indices]
      nearest_grid = nearest_grid.reset_index(drop = True).drop(columns = 'cell')
      nearest_grid['represented_cell'] = range(n_cells)

      # find linear neighbours
      max_cells = np.product(blocks) # = n_cells
      cell_ids = range(n_cells)

      cell_z = []
      for cell in cell_ids:
            # TODO kein nan appenden, einfach continue (überprüfen ob das stimmt, aber ich seh keinen sinn ein None zurück zu geben)
            if cell < 0 or cell > max_cells-1:
                  cell_z.append(np.array([np.nan] * len(blocks)))
            else:
                  coord = []
                  for i in range(len(blocks)):
                        coord.append(cell % blocks[i])
                        cell = cell // blocks[i]
                  cell_z.append(np.array(coord) + 1)

      cell_z = np.array(cell_z)
      if cm_conv_diag:
            combs = _cartesian_product_efficient([[-1, 0, 1]] * len(blocks))
            combs = combs[(combs > 0).any(axis = 1)]
      else:
            combs = np.identity(dim)
      
      nbs = []
      for i in cell_ids:
            x = cell_z[i]
            if ((x == blocks) | (x == 1)).all():
                  nbs.append(np.array([np.nan]))
                  continue
            inner_nb = []
            for comb in combs:
                  z = x + comb
                  # z to cell
                  if (z > blocks).any() or (z < 1).any():
                        inner_nb.append(np.array([np.nan]))
                        continue
                  z = z - 1
                  dim_prod = np.hstack([[1], np.cumprod(blocks[:-1])])
                  succ = (dim_prod * z).sum()
                  nb = np.array([2 * cell_ids[i] - succ, cell_ids[i], succ])
                  if (nb >= 0).all() and (nb <= max_cells - 1).all():
                        inner_nb.append(nb)
                  else:
                        inner_nb.append(np.array([np.nan]))
            nbs.extend(inner_nb)

      #for inner_nb in nbs
      nbs = np.array([x for x in nbs if not np.isnan(x).all()])
      if cm_conv_diag and len(nbs) != 0:
            nbs.sort(axis = 1)
            nbs = np.unique(nbs, axis = 0)
            ind = np.array([x[:2] for x in nbs])
            ind = np.lexsort((ind[:, 0], ind[: , 1]))
            nbs = nbs[ind]

      convexity_counter = []
      for nb_block in nbs:
            x = nearest_grid[nearest_grid.represented_cell.isin(nb_block)]
            yvals = x.sort_values('represented_cell').y.values
            # 0. convex.hard, 1. concave.hard, 2. convex.soft, 3. concave.soft
            counter = np.array([False] * 4)
            if yvals[1] > yvals[[0, 2]].mean():
                  counter[3] = True
                  if yvals[1] > yvals[[0, 2]].max():
                        counter[1] = True
            elif yvals[1] < yvals[[0, 2]].mean():
                  counter[2] = True
                  if yvals[1] < yvals[[0, 2]].min():
                        counter[0] = True
            convexity_counter.append(counter)

      convexity_counter = np.array(convexity_counter).mean(axis = 0)

      return {
            'cm_conv.convex.hard': convexity_counter[0], 
            'cm_conv.concave.hard': convexity_counter[1],
            'cm_conv.convex.soft': convexity_counter[2],
            'cm_conv.concave.soft': convexity_counter[3],
            'cm_conv.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_cm_grad(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      lower_bound: Union[List[float], float],
      upper_bound: Union[List[float], float],
      blocks: Optional[Union[List[int], np.ndarray, int]] = None,
      minimize: bool = True) -> Dict[str, Union[int, float]]:
      """Calculation of Cell Mapping Gradient Homogeneity features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      lower_bound : Union[List[float], float]
          Lower bound of variables of the decision space.
      upper_bound : Union[List[float], float]
          Upper bound of variables of the decision space.
      blocks : Optional[Union[List[int], np.ndarray, int]], optional
          Number of blocks per dimension, by default None.
      minimize : bool, optional
          Indicator whether the objective function should be minimized or maximized, by default True.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)
      dim = X.shape[1]
      blocks = _check_blocks_variable(X, dim, blocks)

      if blocks.min() <= 2:
            raise ValueError('The cell grad features can only be computed when all dimensions have more than 2 cells.')

      init = X.copy()
      init['y'] = y if minimize == True else -1 * y
      init['cell'], _ = _create_blocks(X, y, lower_bound, upper_bound, blocks)

      grad_homo = []
      for _, cell in init.groupby(['cell']):
            n_obs = cell.shape[0]
            funvals = cell['y'].values
            cell = cell.to_numpy()
            if n_obs > 2:
                  nbrs = NearestNeighbors(n_neighbors = 2, algorithm='kd_tree').fit(cell[:, :dim])
                  distance, nn = nbrs.kneighbors(cell[:, :dim])
                  norm_vectors = []
                  for i in range(n_obs):
                        if distance[i, 1] == 0:
                              norm_vectors.append(np.array([0] * dim))
                        else:
                              nearest = nn[i, 1]
                              mult = -1 if funvals[i] > funvals[nearest] else 1
                              norm_vectors.append((cell[nearest, :dim] - cell[i, :dim])/distance[i, 1] * mult)
                  
                  norm_vectors = np.array(norm_vectors)
                  grad_homo.append(np.sqrt(np.sum(np.sum(norm_vectors, axis = 0) ** 2)) / n_obs)
            else:
                  grad_homo.append(np.nan)

      return {
            'cm_grad.mean': np.nanmean(grad_homo),
            'cm_grad.sd': np.nanstd(grad_homo, ddof = 1), 
            'cm_grad.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_ela_conv(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      f: Callable[[List[float]], float],
      ela_conv_nsample: int = 1000,
      ela_conv_threshold: float = 1e-10) -> Dict[str, Union[int, float]]:
      """Calculation of ELA Convexity features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      f : Callable[[List[float]], float]
          Objective function to be optimized.
      ela_conv_nsample : int, optional
          Number of samples that are drawn for calculating the convexity features, by default 1000.
      ela_conv_threshold : float, optional
          Threshold of the linearity, i.e., the tolerance to/deviation from perfect linearity,
          in order to still be considered linear, by default 1e-10.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.
      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)
      delta = []
      nfev = 0
      for _ in range(ela_conv_nsample):
            i = np.random.randint(low = 0, high = X.shape[0], size = 2)
            wt = np.random.uniform(size = 1)[0]
            wt = np.array([wt, 1 - wt])
            xn = np.matmul(wt, X.iloc[i].to_numpy())
            delta.append(f(xn) - np.matmul(y.iloc[i], wt))
            nfev += 1
            
      delta = np.array(delta)

      return {
            'ela_conv.conv_prob': np.nanmean(delta < -ela_conv_threshold), 
            'ela_conv.lin_prob': np.nanmean(np.abs(delta) <= ela_conv_threshold),
            'ela_conv.lin_dev_orig': np.nanmean(delta), 
            'ela_conv.lin_dev_abs': np.nanmean(np.abs(delta)), 
            'ela_conv.additional_function_eval': nfev,
            'ela_conv.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

# TODO mda missing
def calculate_ela_level(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      ela_level_quantiles: List[float] = [0.1, 0.25, 0.5],
      interface_mda_from_R: bool = False,
      ela_level_resample_iterations: int = 10) -> Dict[str, Union[int, float]]:
      """Calculation of ELA Levelset features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      ela_level_quantiles : List[float], optional
          Cutpoints (quantiles of the objective values) for splitting the objective space, by default [0.1, 0.25, 0.5].
      interface_mda_from_R : bool, optional
          Indicator whether to interface missing functionality from R, by default False.
      ela_level_resample_iterations : int, optional
          Number of iterations of the resampling method, by default 10.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)
      mda_mmce = []
      # TODO Interface to R here
      if interface_mda_from_R:
            #mda = _interface_mda()
            
            raise NotImplementedError('MDA is not implemented yet.')
            

      lda_mmce = []
      qda_mmce = []

      for prob in ela_level_quantiles:
            y_quant = np.quantile(y, prob)
            y_class = (y < y_quant)
            if y_class.sum() < ela_level_resample_iterations:
                  raise Exception(f'There are too few observation in case of quantile {prob} to perform resampling with {ela_level_resample_iterations} folds.')
            #data['class'] = [int(x) + 1 for x in data['class']]

            kf = StratifiedKFold(n_splits = ela_level_resample_iterations)
            
            lda_mmce_prob = []
            qda_mmce_prob = []
            if interface_mda_from_R:
                  mda_mmce_prob = []
            for train_index, test_index in kf.split(X, y_class):
                  lda = LinearDiscriminantAnalysis()
                  lda.fit(X.iloc[train_index], y_class[train_index])
                  lda_mmce_prob.append((y_class[test_index].values != lda.predict(X.iloc[test_index])).mean())

                  qda = QuadraticDiscriminantAnalysis()
                  qda.fit(X.iloc[train_index], y_class[train_index])
                  qda_mmce_prob.append((y_class[test_index].values != qda.predict(X.iloc[test_index])).mean())

                  # TODO Calculation of MDA here
                  if interface_mda_from_R:
                        mda_mmce_prob.append(0)

            lda_mmce.append(np.array(lda_mmce_prob).mean())
            qda_mmce.append(np.array(qda_mmce_prob).mean())
            if interface_mda_from_R:
                  mda_mmce.append(np.array(mda_mmce_prob).mean())

      lda_qda = np.array([lda_mmce[i]/qda_mmce[i] for i in range(len(ela_level_quantiles))])
      if interface_mda_from_R:
            lda_mda = np.array([lda_mmce[i]/mda_mmce[i] for i in range(len(ela_level_quantiles))])
            qda_mda = np.array([lda_mmce[i]/mda_mmce[i] for i in range(len(ela_level_quantiles))])

      result = {}
      for i in range(len(ela_level_quantiles)):
            name = 'ela_level.mmce_lda_{:.0f}'.format(ela_level_quantiles[i]*100)
            result[name] = lda_mmce[i]

            name = 'ela_level.mmce_qda_{:.0f}'.format(ela_level_quantiles[i]*100)
            result[name] = qda_mmce[i]

            name = 'ela_level.lda_qda_{:.0f}'.format(ela_level_quantiles[i]*100)
            result[name] = lda_qda[i]
            
            if interface_mda_from_R:
                  name = 'ela_level.mmce_mda_{:.0f}'.format(ela_level_quantiles[i]*100)
                  result[name] = mda_mmce[i]

                  name = 'ela_level.lda_mda_{:.0f}'.format(ela_level_quantiles[i]*100)
                  result[name] = lda_mda[i]

                  name = 'ela_level.qda_mda_{:.0f}'.format(ela_level_quantiles[i]*100)
                  result[name] = qda_mda[i]

      result['ela_level.costs_runtime'] = timedelta(seconds=time.monotonic() - start_time).total_seconds()
      return result

def calculate_ela_curvate(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      f: Callable[[List[float]], float],
      dim: int,
      lower_bound: Union[List[float], float],
      upper_bound: Union[List[float], float],
      sample_size_factor: int = 100,
      delta: float = 10**-4,
      eps: float = 10**-4,
      zero_tol: float = np.sqrt(np.nextafter(0, 1)/70**-7),
      r: int = 4,
      v: int = 2,
      seed: Optional[int] = None) -> Dict[str, Union[int, float]]:
      """Calculation of ELA Curvature features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      f : Callable[[List[float]], float]
          Objective function to be optimized.
      dim : int
          Dimensionality of the decision space.
      lower_bound : Union[List[float], float]
          Lower bound of variables of the decision space.
      upper_bound : Union[List[float], float]
          Upper bound of variables of the decision space.
      sample_size_factor : int, optional
          Factor which determines the sample size by `sample_size_factor * dim`, by default 100.
      delta : float, optional
          Parameter used to approximate the gradient and hessian.
          See `grad` and `hessian` of the R-package numDeriv for more details, by default 10**-4.
      eps : float, optional
          Parameter used to approximate the gradient and hessian.
          See `grad` and `hessian` of the R-package numDeriv for more details, by default 10**-4.
      zero_tol : float, optional
          Parameter used to approximate the gradient and hessian.
          See `grad` and `hessian` of the R-package numDeriv for more details, by default np.sqrt(np.nextafter(0, 1)/70**-7).
      r : int, optional
          Parameter used to approximate the gradient and hessian.
          See `grad` and `hessian` of the R-package numDeriv for more details, by default 4.
      v : int, optional
          Parameter used to approximate the gradient and hessian.
          See `grad` and `hessian` of the R-package numDeriv for more details, by default 2.
      seed : Optional[int], optional
          Seed for reproducability, by default None.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      """      
      start_time = time.monotonic()

      X, y = _validate_variable_types(X, y)
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)
      if seed is not None:
            np.random.seed(seed)

      N = sample_size_factor * dim
      if X.shape[0] < N:
            N = X.shape[0]

      def decorator(f, x):
            decorator.fvals += 1
            return f(x)

      decorator.fvals = 0
      original_f = f
      f = partial(decorator, original_f)

      wfunc = partial(_calculate_num_derivate, f, lower_bound, upper_bound, delta, eps, zero_tol, r, v)
      derivs = X.sample(N).apply(lambda x: wfunc(x.values), axis = 1)
      derivs = np.array([x for x in derivs]).reshape(3, derivs.shape[0])
      
      return {
            'ela_curv.grad_norm.min': np.nanmin(derivs[0]),
            'ela_curv.grad_norm.lq': np.nanquantile(derivs[0], 0.25),
            'ela_curv.grad_norm.mean': np.nanmean(derivs[0]),
            'ela_curv.grad_norm.med': np.nanmedian(derivs[0]),
            'ela_curv.grad_norm.uq': np.nanquantile(derivs[0], 0.75),
            'ela_curv.grad_norm.max': np.nanmax(derivs[0]),
            'ela_curv.grad_norm.sd': np.nanstd(derivs[0], ddof = 1),
            'ela_curv.grad_norm.nas': np.mean(np.isnan(derivs[0])),
            'ela_curv.grad_scale.min': np.nanmin(derivs[1]),
            'ela_curv.grad_scale.lq': np.nanquantile(derivs[1], 0.25),
            'ela_curv.grad_scale.mean': np.nanmean(derivs[1]),
            'ela_curv.grad_scale.med': np.nanmedian(derivs[1]),
            'ela_curv.grad_scale.uq': np.nanquantile(derivs[1], 0.75),
            'ela_curv.grad_scale.max': np.nanmax(derivs[1]),
            'ela_curv.grad_scale.sd': np.nanstd(derivs[1], ddof = 1),
            'ela_curv.grad_scale.nas': np.mean(np.isnan(derivs[1])),
            'ela_curv.hessian_cond.min': np.nanmin(derivs[2]),
            'ela_curv.hessian_cond.lq': np.nanquantile(derivs[2], 0.25),
            'ela_curv.hessian_cond.mean': np.nanmean(derivs[2]),
            'ela_curv.hessian_cond.med': np.nanmedian(derivs[2]),
            'ela_curv.hessian_cond.uq': np.nanquantile(derivs[2], 0.75),
            'ela_curv.hessian_cond.max': np.nanmax(derivs[2]),
            'ela_curv.hessian_cond.sd': np.nanstd(derivs[2], ddof = 1),
            'ela_curv.hessian_cond.nas': np.mean(np.isnan(derivs[2])),
            'ela_curv.costs_fun_evals:': decorator.fvals,
            'ela_curv.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()
      }

def calculate_ela_local(
      X: Union[pd.DataFrame, np.ndarray, List[List[float]]],
      y: Union[pd.Series, np.ndarray, List[float]],
      f: Callable[[List[float]], float],
      dim: int,
      lower_bound: Union[List[float], float],
      upper_bound: Union[List[float], float],
      minimize: bool = True,
      ela_local_local_searches_factor: int = 50,
      ela_local_optim_method: str = 'L-BFGS-B',
      ela_local_clust_method: str = 'single',
      seed: Optional[int] = None,
      **minimizer_kwargs) -> Dict[str, Union[int, float]]:
      """Calculation of ELA Local Search features, similar to the R-package `flacco`.

      Parameters
      ----------
      X : Union[pd.DataFrame, np.ndarray, List[List[float]]]
          A collection-like object which contains a sample of the decision space.
          Can be created with `sampling.create_initial_sample`.
      y : Union[pd.Series, np.ndarray, List[float]]
          A list-like object which contains the respective objective values of `X`.
      f : Callable[[List[float]], float]
          Objective function to be optimized.
      dim : int
          Dimensionality of the decision space.
      lower_bound : Union[List[float], float]
          Lower bound of variables of the decision space.
      upper_bound : Union[List[float], float]
          Upper bound of variables of the decision space.
      minimize : bool, optional
          Indicator whether the objective function should be minimized or maximized, by default True.
      ela_local_local_searches_factor : int, optional
          Factor which determines the number of local searches by
          `ela_local_local_searches_factor * dim`, by default 50.
      ela_local_optim_method : str, optional
          Type of solver. Any of `scipy.optimize.minimize` can be used, by default 'L-BFGS-B'.
      ela_local_clust_method : str, optional
          Hierarchical clustering method to use, by default 'single'.
      seed : Optional[int], optional
          Seed for reproducability, by default None.

      Returns
      -------
      Dict[str, Union[int, float]]
          Dictionary consisting of the calculated features.

      """      
      start_time = time.monotonic()
      X, y = _validate_variable_types(X, y)
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)
      N = ela_local_local_searches_factor * dim
      if not minimize:
            y = y * -1
            original_f = f
            f = lambda x: -1 * original_f(x)

      if X.shape[0] < N:
            raise ValueError(f'X contains less then the required {N} (= dim * ela_local_local_searches_factor) starting points')
      if seed is not None:
            np.random.seed(seed)
      
      bounds = list(zip(lower_bound, upper_bound))
      x_opts = []
      fes = []

      for _, row in X.sample(N, replace = False).iterrows():
            opt_result = scipy_minimize(f, row.values, method = ela_local_optim_method, bounds = bounds, **minimizer_kwargs)
            x_opts.append(opt_result.x)
            fes.append(opt_result.nfev)
      
      x_opts = np.array(x_opts)
      fes = np.array(fes)

      cl = linkage(x_opts, method = ela_local_clust_method)
      nodes = _order_cluster_tree(cl)

      heights = np.array([x.dist for x in nodes])
      c_assign = cut_tree(cl, height = np.quantile(heights, 0.1)).flatten()
      clust_sizes = np.array([(c_assign == x).sum()/len(np.unique(c_assign)) for x in np.unique(c_assign)])
      c_centers = []
      center_fvals = []
      for i in np.unique(c_assign):
            center = x_opts[c_assign == i].mean(axis = 0)
            c_centers.append(center)
            center_fvals.append(f(center))
      c_centers = np.array(c_centers)
      center_fvals = np.array(center_fvals)

      center_best_idx = center_fvals == center_fvals.min()
      center_worst_idx = center_fvals == center_fvals.max()
      return {
            'ela_local.n_loc_opt.abs': len(c_centers),
            'ela_local.n_loc_opt.rel': len(c_centers)/N,
            'ela_local.best2mean_contr.orig': center_fvals.min()/center_fvals.mean(),
            'ela_local.best2mean_contr.ratio': (center_fvals.mean() - center_fvals.min())/ (center_fvals.max() - center_fvals.min()),
            'ela_local.basin_sizes.avg_best': clust_sizes[center_best_idx].mean(),
            'ela_local.basin_sizes.avg_non_best':  0 if len(clust_sizes[~center_best_idx]) == 0 else clust_sizes[~center_best_idx].mean(),
            'ela_local.basin_sizes.avg_worst': clust_sizes[center_worst_idx].mean(),
            'ela_local.fun_evals.min': fes.min(),
            'ela_local.fun_evals.lq': np.quantile(fes, 0.25),
            'ela_local.fun_evals.mean': fes.mean(),
            'ela_local.fun_evals.median': np.median(fes),
            'ela_local.fun_evals.uq': np.quantile(fes, 0.75),
            'ela_local.fun_evals.max': fes.max(),
            'ela_local.fun_evals.sd': fes.std(ddof = 1),
            'ela_local.additional_function_eval': fes.sum() + len(c_centers),
            'ela_local.costs_runtime': timedelta(seconds=time.monotonic() - start_time).total_seconds()

      }
