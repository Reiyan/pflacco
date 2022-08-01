import numpy as np
import math
import os
import pandas as pd
import itertools

from copy import copy
from pyDOE import lhs
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from scipy.stats import levy
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import entropy, moment
from SALib.analyze import sobol
from SALib.sample import sobol_sequence

from bbob import bbobbenchmarks

from utils import *

allowed_ela_features = ['limo', 'ela_meta', 'ela_level', 'ela_distr', 'disp', 'ic', 'pca', 'nbc', 'fd']


def _levy_random_walk(x):
      
      vec = np.random.normal(0, 1, len(x))
      norm_vec = vec/(np.sqrt((vec ** 2).sum()))
      step_size = levy.rvs(size = 1, loc = 0, scale = 10**-3)

      return x + step_size * norm_vec 


def _create_local_search_sample(f, dim, lower_bound, upper_bound, n_runs = 100, budget_factor_per_run=1000, method = 'L-BFGS-B', minimize = True, seed = None, x0 = None):
      if not isinstance(lower_bound, list) and type(lower_bound) is not np.ndarray:
            lower_bound = np.array([lower_bound] * dim)
      if isinstance(lower_bound, list):
            lower_bound = np.array(lower_bound)
      if not isinstance(upper_bound, list) and type(upper_bound) is not np.ndarray:
            upper_bound = np.array([upper_bound] * dim)
      if isinstance(upper_bound, list):
            upper_bound = np.array(upper_bound)
      if len(lower_bound) != dim or len(upper_bound) != dim:
            raise Exception('Length of lower-/upperbound is not the same as the problem dimension')
      if not (lower_bound < upper_bound).all():
            raise Exception('Not all elements of lower bound are smaller than upper bound')

      if not minimize:
            original_f = f
            f = lambda x: -1 * original_f(x)
      if seed is not None:
            np.random.seed(seed)

      minimizer_kwargs = {
            'maxfun': budget_factor_per_run*dim,
            'ftol': 1e-8
            
      }
      bounds = list(zip(lower_bound, upper_bound))
      result = []
      for _ in range(n_runs):
            x0 = np.random.uniform(low = lower_bound, high = upper_bound, size = dim)
            opt_result = scipy_minimize(f, x0, method = method, bounds = bounds, options = minimizer_kwargs)
            result.append(np.append(opt_result.x, [opt_result.fun]))

      return np.array(result)

def create_initial_sample(dim, n = None, sample_coefficient = 50, lower_bound = 0, upper_bound = 1, sample_type = 'lhs'):
      if sample_type not in ['lhs', 'random', 'sobol']:
            raise Exception('Unknown sample type selected. Valid options are "lhs" and "random"')

      if not isinstance(lower_bound, list) and type(lower_bound) is not np.ndarray:
            lower_bound = np.array([lower_bound] * dim)
      if isinstance(lower_bound, list):
            lower_bound = np.array(lower_bound)
      
      if not isinstance(upper_bound, list) and type(upper_bound) is not np.ndarray:
            upper_bound = np.array([upper_bound] * dim)
      if isinstance(upper_bound, list):
            upper_bound = np.array(upper_bound)

      if len(lower_bound) != dim or len(upper_bound) != dim:
            raise Exception('Length of lower-/upperbound is not the same as the problem dimension')
      
      if not (lower_bound < upper_bound).all():
            raise Exception('Not all elements of lower bound are smaller than upper bound')

      if n is None:
            n = dim * sample_coefficient

      if sample_type == 'lhs':
            X = lhs(dim, samples = n)
      elif sample_type == 'sobol':
            X = sobol_sequence.sample(n, dim)
      else:
            X = np.random.rand(n, dim)
      
      X = X * (upper_bound - lower_bound) + lower_bound
      colnames = ['x' + str(x) for x in range(dim)]
      
      return pd.DataFrame(X, columns = colnames)

def _create_blocks(X, y, lower_bound = None, upper_bound = None, blocks = 1):
      if X.shape[0] != len(y):
            raise Exception('Dataframe X and array y must provide the same amount of observation.')

      dim = X.shape[1]

      if not isinstance(blocks, list) and type(blocks) is not np.ndarray:
            blocks = np.array([blocks] * dim)
      if isinstance(blocks, list):
            blocks = np.array(blocks)
      if len(blocks) != dim:
            raise Exception('The provided value for "block" does not have the same length as the dimensionality of X.')
      
      if lower_bound is None:
            lower_bound = X.min().to_numpy()
      if not isinstance(lower_bound, list) and type(lower_bound) is not np.ndarray:
            lower_bound = np.array([lower_bound] * dim)
      if isinstance(lower_bound, list):
            lower_bound = np.array(lower_bound)
      if len(lower_bound) != dim:
            raise Exception('The provided value for "lower_bound" does not have the same length as the dimensionality of X.')
      
      if upper_bound is None:
            upper_bound = X.max().to_numpy()
      if not isinstance(upper_bound, list) and type(upper_bound) is not np.ndarray:
            upper_bound = np.array([upper_bound] * dim)
      if isinstance(upper_bound, list):
            upper_bound = np.array(upper_bound)
      if len(upper_bound) != dim:
            raise Exception('The provided value for "upper_bound" does not have the same length as the dimensionality of X.')

      block_widths = (upper_bound - lower_bound)/blocks
      cp = np.cumprod(np.insert(blocks, 0, 1))

      cell_ids = []
      for idx, row in X.iterrows():
            cid = [cp[ndim] * np.floor((row[ndim] - lower_bound[ndim]) / block_widths[ndim]) for ndim in range(X.shape[1])]
            cell_ids.append((cid - cp[:-1] * (row == upper_bound)).sum()) 
      cell_ids = np.array(cell_ids)


      n_centers = []
      for idx in range(len(blocks)):
            tmp = np.linspace(lower_bound[idx], upper_bound[idx], blocks[idx] + 1)
            n_centers.append((tmp[1:] + tmp[:-1])/2)
      # Artificial complicated sorting to replicate output from expand.grid in R
      c_centers = pd.DataFrame(np.array([x for x in itertools.product(*[y for y in n_centers])]))
      c_centers = c_centers.sort_values(list(reversed(c_centers.columns))).to_numpy()

      return cell_ids, c_centers
                  

def calculate_ela_meta(X, y):
      if not isinstance(y, pd.Series) and (isinstance(y, np.ndarray) or isinstance(y, list)):
            y = pd.Series(y, name = 'y')
      if not isinstance(X, pd.DataFrame):
            raise Exception('Samples of the decision space have to be stored in pandas dataframe')

      # Create liner model and calculate lm features
      model = linear_model.LinearRegression()
      model.fit(X, y)
      pred = model.predict(X)

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
      pred = model.predict(X_interact)
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
      pred = model.predict(X_interact)
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
            'ela_meta.quad_w_interact.adj_r2': quad_w_interact_adj_r2
      }

def calculate_pca(X, y, prop_cov_x = 0.9, prop_cor_x = 0.9, prop_cov_init = 0.9, prop_cor_init = 0.9):
      if not isinstance(X, pd.DataFrame):
            raise Exception('Samples of the decision space have to be stored in pandas dataframe')

      if not isinstance(y, pd.Series) and (isinstance(y, np.ndarray) or isinstance(y, list)):
            y = pd.Series(y, name = 'y')

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
          'pca.expl_var_PC1.cor_init': expl_var_pc1_cor_init
      }
# TODO rename nbc_fask_k to nbc_fast_k 
def calculate_nbc(X, y, nbc_fask_k = 0.05, nbc_dist_tie_breaker = 'sample', minimize = True):
      if not isinstance(y, pd.Series) and (isinstance(y, np.ndarray) or isinstance(y, list)):
            y = pd.Series(y, name = 'y')

      if nbc_fask_k < 1:
            nbc_fask_k = math.ceil(nbc_fask_k * X.shape[0])
      if nbc_fask_k < 0 or nbc_fask_k > X.shape[0]:
            raise Exception(f'[{nbc_fask_k}] of "nbc_fask_k" does not lie in the interval [0,n] where n is the number of observations.')
      if minimize == False:
            y = y * -1

      nbrs = NearestNeighbors(n_neighbors = nbc_fask_k, algorithm='kd_tree').fit(X)
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
                        if nbc_dist_tie_breaker == 'sample':
                              j = np.random.choice(d[d == d.min()].index)
                              results.append([idx, ind_alt[j], d[j]])
                        else:
                              #TODO welche anderen Tiebreaker methoden gibt es? es gibt noch first und last
                              raise Exception('Currently, the only available tie breaker method is "sample"')

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
            'nbc.nb_fitness.cor': nb_stats['toMe_count'].corr(nb_stats['fitness'])
      }

def calculate_dispersion(X, y, disp_quantiles = [0.02, 0.05, 0.1, 0.25], disp_dist_method = 'euclidean', disp_dist_p = 2, minimize = True):
      if not isinstance(y, pd.Series) and (isinstance(y, np.ndarray) or isinstance(y, list)):
            y = pd.Series(y, name = 'y')

      if minimize == False:
            y = y * -1

      quantiles = np.quantile(y, disp_quantiles)
      
      idx_in_quantiles = {}
      for idx, quantile in enumerate(quantiles):
            idx_in_quantiles[str(disp_quantiles[idx])] = [x for x in y[y < quantile].index]

      dists = {}
      # Parameter p only will apply if method is 'minkowski'. Otherwise it will be ignored by scipy
      for quantile in idx_in_quantiles:
            idx_in_quantiles[quantile]
            dists[quantile] = squareform(pdist(X.iloc[idx_in_quantiles[quantile]], metric = disp_dist_method, p = disp_dist_p))

      dist_full_sample = squareform(pdist(X, metric = disp_dist_method))

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

      return result

def calculate_information_content(X, y, ic_sorting = 'nn', ic_nn_neighborhood = 20, ic_nn_start = None,\
      ic_epsilon = np.insert(10 ** np.linspace(start = -5, stop = 15, num = 1000), 0, 0), ic_settling_sensitivity = 0.05, ic_info_sensitivity = 0.5, ic_seed = None):
      if not isinstance(y, pd.Series) and (isinstance(y, np.ndarray) or isinstance(y, list)):
            y = pd.Series(y, name = 'y')

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

            
      if ic_seed is not None and isinstance(ic_seed, int):
            np.random.seed(ic_seed)

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
            'ic.m0': m0[0]
      }

def calculate_ela_distribution(X, y, ela_distr_smoothing_bandwith = 'SJ', ela_distr_modemass_threshold = 0.01, ela_distr_skewness_type = 3, ela_distr_kurtosis_type = 3):
      if not isinstance(y, pd.Series) and (isinstance(y, np.ndarray) or isinstance(y, list)):
            y = pd.Series(y, name = 'y')
      if ela_distr_skewness_type not in range(1,4):
            raise Exception('Skewness type must be an integer and in the intervall [1,3]')
      if ela_distr_kurtosis_type not in range(1,4):
            raise Exception('Kurtosis type must be an integer and in the intervall [1,3]')
      
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
            'ela_distr.number_of_peaks': n_peaks
      }
            
def calculate_limo(X, y, lower_bound, upper_bound, blocks = None):
      if not isinstance(y, pd.Series) and (isinstance(y, np.ndarray) or isinstance(y, list)):
            y = pd.Series(y, name = 'y')

      if blocks is None:
            blocks = _determine_max_n_blocks(X)

      dims = X.shape[1]

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

      if blocks > 1:
            result['limo.length_sd'] = np.std(length_coeff_vector, ddof = 1)
            result['limo.cor'] = (np.sum(cor_unscaled) - np.sum(np.diag(cor_unscaled))) / (np.ma.count(cor_unscaled) - len(np.diag(cor_unscaled)))
            result['limo.cor_norm'] = (np.sum(cor_scaled) - np.sum(np.diag(cor_scaled))) / (np.ma.count(cor_scaled) - len(np.diag(cor_scaled)))
            result['limo.ratio_sd'] = np.std(coeff_ratio, ddof = 1)
            result['limo.sd_ratio_reg'] = np.max(sds_unscaled) / np.min(sds_unscaled)
            result['limo.sd_ratio_norm'] = np.max(sds_scaled) / np.min(sds_scaled)
            result['limo.sd_mean_reg'] = np.mean(sds_unscaled)
            result['limo.sd_mean_norm'] = np.mean(sds_scaled)

      return result

def calculate_cm_angle(X, y, lower_bound, upper_bound, blocks = None, minimize = True):
      dim = X.shape[1]
      if blocks is None:
            blocks = _determine_max_n_blocks(X)
      if not isinstance(blocks, list) and type(blocks) is not np.ndarray:
            blocks = np.array([blocks] * dim)
      elif isinstance(blocks, list):
            blocks = np.array(blocks)
      if len(blocks) != dim:
            raise Exception('The provided value for "block" does not have the same length as the dimensionality of X.')

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
      }

def calculate_cm_conv(X, y, lower_bound, upper_bound, blocks = None, minimize = True, cm_conv_diag = False,  cm_conv_fast_k = 0.05):
      dim = X.shape[1]
      if blocks is None:
            blocks = max(_determine_max_n_blocks(X), 2)
      if not isinstance(blocks, list) and type(blocks) is not np.ndarray:
            blocks = np.array([blocks] * dim)
      elif isinstance(blocks, list):
            blocks = np.array(blocks)
      if len(blocks) != dim:
            raise Exception('The provided value for "block" does not have the same length as the dimensionality of X.')
      if blocks.min() <= 2:
            raise Exception('The cell convexity features can only be computed when all dimensions have more than 2 cells.')
      if cm_conv_fast_k < 0 or cm_conv_fast_k > X.shape[0]:
            raise Exception('cm_conv_fast_k must be in the interval [0, n] where n is the number of observations in X.')

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
            x = x[x >= 0][0] if len([x[x >= 0]]) > 0 else None
            nearest_grid_indices.append(x)

      nearest_grid = init.iloc[nearest_grid_indices]

      # in case none of the nearest observations is a non-cell-center
      all_centers = np.all(indices < 0, axis = 1)
      if all_centers.any():
            n_ctr = all_centers.sum()
            nbrs_backup = NearestNeighbors(n_neighbors = n_ctr + 1, algorithm='kd_tree').fit(np.vstack([cell_centers[all_centers], X.to_numpy()]))
            _, backup_indices = nbrs_backup.kneighbors(np.vstack([cell_centers[all_centers], X.to_numpy()]))
            backup_indices = backup_indices[range(n_ctr), 1:] - n_ctr
            if len(backup_indices[backup_indices >= 0]) > 0:
                  backup_indices = backup_indices[backup_indices >= 0][0]
                  nearest_grid[all_centers] = init.iloc[backup_indices].values
            else:
                  raise Exception('Could not determine backup neighbor in cm_conv')
      nearest_grid = nearest_grid.reset_index(drop = True)
      nearest_grid['represented_cell'] = range(n_cells)

      # find linear neighbours
      max_cells = np.product(blocks) # = n_cells
      cell_ids = range(n_cells)

      tmp = []
      for cell in cell_ids:
            if cell < 0 or cell > max_cells-1:
                  tmp.append(None)
            else:
                  cell = cell - 1
                  coord = []
                  for i in range(len(blocks)):
                        coord.append(cell % blocks[i])
      print('test')
      #TODO not finished


def calculate_cm_grad(X, y, lower_bound, upper_bound, blocks = None, minimize = True, cm_conv_diag = False,  cm_conv_fast_k = 0.05):
      dim = X.shape[1]
      if blocks is None:
            blocks = max(_determine_max_n_blocks(X), 2)
      if not isinstance(blocks, list) and type(blocks) is not np.ndarray:
            blocks = np.array([blocks] * dim)
      elif isinstance(blocks, list):
            blocks = np.array(blocks)
      if len(blocks) != dim:
            raise Exception('The provided value for "block" does not have the same length as the dimensionality of X.')
      if blocks.min() <= 2:
            raise Exception('The cell convexity features can only be computed when all dimensions have more than 2 cells.')
      if cm_conv_fast_k < 0 or cm_conv_fast_k > X.shape[0]:
            raise Exception('cm_conv_fast_k must be in the interval [0, n] where n is the number of observations in X.')

      
      init = X.copy()
      init['y'] = y if minimize == True else -1 * y
      init['cell'], cell_centers = _create_blocks(X, y, lower_bound, upper_bound, blocks)

      grad_homo = []
      for cname, cell in init.groupby(['cell']):
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



      print('test')

      return {
            'cm_grad.mean': np.nanmean(grad_homo),
            'cm_grad.sd': np.nanstd(grad_homo, ddof = 1)
      }


def calculate_ela_conv(X, y, fun, lower_bound, upper_bound, blocks = None, minimize = True, ela_conv_nsample = 1000, ela_conv_threshold = 1e-10):
      delta = []
      for _ in range(ela_conv_nsample):
            i = np.random.randint(low = 0, high = X.shape[0], size = 2)
            wt = np.random.uniform(size = 1)[0]
            wt = np.array([wt, 1 - wt])
            xn = np.matmul(wt, X.iloc[i].to_numpy())
            delta.append(fun(xn) - np.matmul(y.iloc[i], wt))
            i = i + 1
      delta = np.array(delta)

      return {
            'ela_conv.conv_prob': np.nanmean(delta < -ela_conv_threshold), 
            'ela_conv.lin_prob': np.nanmean(np.abs(delta) <= ela_conv_threshold),
            'ela_conv.lin_dev_orig': np.nanmean(delta), 
            'ela_conv.lin_dev_abs': np.nanmean(np.abs(delta))
      }

def calculate_ela_level(X, y, ela_level_quantiles = [0.1, 0.25, 0.5], ela_level_classif_methods = ['lda', 'qda', 'mda'], ela_level_resample_iterations = 10, ela_level_resample_method = 'CV'):
      if not isinstance(y, pd.Series) and (isinstance(y, np.ndarray) or isinstance(y, list)):
            y = pd.Series(y, name = 'y')

      result = {}
      for prob in ela_level_quantiles:
            y_quant = np.quantile(y, prob)
            #data = copy(X)
            y_class = (y < y_quant)
            if y_class.sum() < ela_level_resample_iterations:
                  raise Exception(f'There are too few observation in case of quantile {prob} to perform resampling with {ela_level_resample_iterations} folds.')
            #data['class'] = [int(x) + 1 for x in data['class']]

            kf = KFold(n_splits = ela_level_resample_iterations)
            
            lda_mmce = []
            qda_mmce = []
            for train_index, test_index in kf.split(X):
                  lda = LinearDiscriminantAnalysis()
                  lda.fit(X.iloc[train_index], y_class[train_index])
                  mmce = accuracy_score(y_class[test_index], lda.predict(X.iloc[test_index]))
                  lda_mmce.append(mmce)

                  qda = QuadraticDiscriminantAnalysis()
                  qda.fit(X.iloc[train_index], y_class[train_index])
                  mmce = accuracy_score(y_class[test_index], qda.predict(X.iloc[test_index]))
                  qda_mmce.append(mmce)
                  print(mmce)
      return None

def calculate_hill_climbing_features(f, dim, lower_bound, upper_bound, n_runs = 100, budget_factor_per_run = 1000, method = 'L-BFGS-B', minimize = True, seed = None, minkowski_p = 2):
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

      opt_result = _create_local_search_sample(f, dim, lower_bound, upper_bound, n_runs = n_runs, budget_factor_per_run=budget_factor_per_run, method = method, minimize = minimize, seed = seed)

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
            'hill_climbing.std_dist_local_to_global': dist_global_local_std
      }
      
def calculate_gradient_features(f, dim, lower_bound, upper_bound, step_size = None, budget_factor_per_dim = 100, seed = None):
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

      if seed is not None:
            np.random.seed(seed)

      if step_size is None:
            step_size = (upper_bound.min() - lower_bound.min())/20

      dd = np.random.choice([0, 1], size = dim)
      bounds = list(zip(lower_bound, upper_bound))
      x = np.array([bounds[x][dd[x]] for x in range(dim)], dtype = 'float64')
      fval = f(x)
      signs = np.array([1 if x == 0 else -1 for x in dd])
      result = [np.append(x, fval)]
      for i in range(budget_factor_per_dim * dim - 1):
            cd = np.random.choice(range(dim))
            if not (x[cd] + signs[cd]*step_size <= bounds[cd][1] and x[cd] + signs[cd]*step_size >= bounds[cd][0]):
                  signs[cd] = signs[cd] * -1
            x[cd] = x[cd] + signs[cd]*step_size
            fval = f(x)
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
            'gradient.g_std': g_dev
      }


def calculate_fitness_distance_correlation(X, y, f_opt = None, proportion_of_best = 1, minimize = True, minkowski_p = 2):
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
            'fitness_distance.fitness_std': y.std(ddof = 1)
      }

            
def calculate_length_scales_features(f, dim, lower_bound, upper_bound, budget_factor_per_dim = 1000, seed = None, minimize = True, sample_size_from_kde = 1000):
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

      if seed is not None:
            np.random.seed(seed)

      bounds = list(zip(lower_bound, upper_bound))

      x = np.random.uniform(lower_bound, upper_bound, dim)
      result = []
      for _ in range(budget_factor_per_dim * (dim ** 2)):
            x = _levy_random_walk(x)
            x = np.array([np.clip(x[i], bounds[i][0], bounds[i][1]) for i in range(len(x))])
            fval = f(x)
            result.append(np.append(x, fval))
      result = np.array(result)
      r_dist = pdist(result[:, :dim])
      r_fval = pdist(result[:, dim].reshape(len(result), 1), metric = 'cityblock')
      r = r_fval/r_dist
      r = r[~np.isnan(r)]
      r = r[~np.isinf(r)]
      kernel = gaussian_kde(r)
      sample = np.random.uniform(low=r.min(), high=r.max(), size = sample_size_from_kde * dim)
      prob = kernel.pdf(sample)
      h_r = entropy(prob, base = 2)
      moment_sample = kernel.resample(sample_size_from_kde*dim).reshape(sample_size_from_kde*dim)
      moments = moment(moment_sample, moment = [2, 3, 4])
      return {
            'length_scale.shanon_entropy': h_r,
            'length_scale.distribution_second_moment': moments[0],
            'length_scale.distribution_third_moment': moments[1],
            'length_scale.distribution_fourth_moment': moments[2]
      }

def calculate_sobol_indices_features(f, dim, lower_bound, upper_bound, sampling_coefficient = 10000, n_bins = 20, min_obs_per_bin_factor = 1.5, seed = None):

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
            'fla_metrics.state_skewness': s_d
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


test = pd.read_csv('pflacco_5D.csv')
#X = test[['x1', 'x2']]
X = test[['x1', 'x2', 'x3', 'x4', 'x5']]
y = test['y']

def test_obj(X):

      return sum(X) ** 2/(np.sqrt(np.abs(np.sum(X))) + 1)

test_obj = bbobbenchmarks.instantiate(1, iinstance=1)[0]
#X = create_initial_sample(5, n = 500, lower_bound = -5, upper_bound = 5, sample_type = 'sobol')
#y = X.apply(lambda x: test_obj(x), axis = 1)
result = calculate_sobol_indices_features(test_obj, 10, -5, 5, seed = 100)
#opt_result = calculate_length_scales_features(test_obj, 10, -5, 5, minimize=True, seed=100, budget_factor_per_dim=10, sample_size_from_kde= 1000)
#print(opt_result)
#result = calculate_fitness_distance_correlation(X, y, proportion_of_best= 1, minimize = False)
print(result)
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


result = calculate_features(X, y, features = ['disp', 'ela_meta', 'ela_distr', 'ic', 'limo', 'nbc'])
result = calculate_ela_level(X, y)
print('test')