import itertools
import numpy as np
import pandas as pd

#from rpy2.robjects.packages import importr, isinstalled

#def _interface_mda():
#    base = importr('base')
#    utils = importr('utils')
#    utils.chooseCRANmirror(ind=1)
#    if not isinstalled('mda'):
#        utils.install_packages('mda')
#    mda = importr('mda')

#    return mda

def _cartesian_product_efficient(arrays):
      arrays = np.array([np.array(x) for x in arrays])
      la = len(arrays)
      dtype = np.find_common_type([a.dtype for a in arrays], [])
      arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
      for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
      return arr.reshape(-1, la)

def _validate_variable_types(X, y):
      if not isinstance(X, pd.DataFrame) and (isinstance(y, np.ndarray) or isinstance(y, list)):
            X = pd.DataFrame(X)
      elif not isinstance(X, pd.DataFrame):
            raise Exception('Unknown format of X. X must be either a Python list, numpy array oder pandas DataFrame')

      X.columns = ['x' + str(x) for x in range(X.shape[1])]
      X = X.reset_index(drop = True)
      if not isinstance(y, pd.Series) and (isinstance(y, np.ndarray) or isinstance(y, list)):
            y = pd.Series(y, name = 'y')
      y = y.reset_index(drop = True)

      if X.shape[0] != len(y):
            raise Exception('pd.DataFrame X and pd.Series y must provide the same amount of observation.')
      return X, y

# Helper function to transform scalar bounds to an N * D array, where D is the dimensionality and N the different lower/upper bounds of the respective dimensions.
def _transform_bounds_to_canonical(dim, lower_bound, upper_bound):
      if lower_bound is None or upper_bound is None:
            raise Exception('No values for either lower bounds or upper bounds where provided. Pflacco is only applicable to box constrainted problems.')
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

      return lower_bound, upper_bound

def _determine_max_n_blocks(X):
      blocks = 1
      while (X.shape[1] ** (blocks + 1)) * 3 < X.shape[0]:
            blocks += 1
      return blocks

def _check_blocks_variable(X, dim, blocks):
      if blocks is None:
            blocks = max(_determine_max_n_blocks(X), 2)
      if not isinstance(blocks, list) and type(blocks) is not np.ndarray:
            blocks = np.array([blocks] * dim)
      elif isinstance(blocks, list):
            blocks = np.array(blocks)
      if len(blocks) != dim:
            raise Exception('The provided value for "block" does not have the same length as the dimensionality of X.')

      return blocks

def _create_blocks(X, y, lower_bound, upper_bound, blocks = None):
      X, y, _validate_variable_types(X, y)
      dim = X.shape[1]
      blocks = _check_blocks_variable(X, dim, blocks)
      lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

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
