import numpy as np
import pandas as pd

from pyDOE import lhs
from SALib.sample import sobol_sequence
from scipy.stats import levy
from scipy.optimize import minimize as scipy_minimize
from typing import List, Optional, Union

from .utils import _transform_bounds_to_canonical

def create_initial_sample(
      dim: int,
      n: Optional[int] = None,
      sample_coefficient: int = 50,
      lower_bound: Union[List[float], float] = 0,
      upper_bound: Union[List[float], float] = 1,
      sample_type: str = 'lhs') -> pd.DataFrame:
      """Sampling of the decision space.

      Parameters
      ----------
      dim : int
          Dimensionality of the search space.
      n : Optional[int], optional
          Fixed number of samples to create. In ELA, this is typically scaled 
          to the dimensionalty of the problem, e.g., ``n=50*dim``, by default None.
      sample_coefficient : int, optional
          Factor which is used to determine the sample size in conjuction
          with the problem dimensionality, by default 50.
      lower_bound : Union[List[float], float], optional
          Lower bound of variables of the decision space, by default 0.
      upper_bound : Union[List[float], float], optional
          Upper bound of variables of the decision space, by default 1.
      sample_type : str, optional
          Type of sampling strategy. Should be one of ('lhs', 'random', 'sobol'), by default 'lhs'.

      Returns
      -------
      pd.DataFrame
          `n` x `dim` shaped Pandas dataframe containing the different samples.
      """      

      if sample_type not in ['lhs', 'random', 'sobol']:
            raise ValueError('Unknown sample type selected. Valid options are "lhs", "sobol", and "random"')

      if not isinstance(lower_bound, list) and type(lower_bound) is not np.ndarray:
            lower_bound = np.array([lower_bound] * dim)
      if isinstance(lower_bound, list):
            lower_bound = np.array(lower_bound)
      
      if not isinstance(upper_bound, list) and type(upper_bound) is not np.ndarray:
            upper_bound = np.array([upper_bound] * dim)
      if isinstance(upper_bound, list):
            upper_bound = np.array(upper_bound)

      if len(lower_bound) != dim or len(upper_bound) != dim:
            raise ValueError('Length of lower-/upper bound is not the same as the problem dimension')
      
      if not (lower_bound < upper_bound).all():
            raise ValueError('Not all elements of lower bound are smaller than upper bound')

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

def _create_local_search_sample(f, dim, lower_bound, upper_bound, n_runs = 100, budget_factor_per_run=1000, method = 'L-BFGS-B', minimize = True, seed = None, x0 = None):
    lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

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
    nfval = 0
    for _ in range(n_runs):
        x0 = np.random.uniform(low = lower_bound, high = upper_bound, size = dim)
        opt_result = scipy_minimize(f, x0, method = method, bounds = bounds, options = minimizer_kwargs)
        result.append(np.append(opt_result.x, [opt_result.fun]))
        nfval += opt_result.nfev

    return np.array(result), nfval

def _levy_random_walk(x, loc = 0, scale = 10**-3):
      
      vec = np.random.normal(0, 1, len(x))
      norm_vec = vec/(np.sqrt((vec ** 2).sum()))
      step_size = levy.rvs(size = 1, loc = loc, scale = scale)

      return x + step_size * norm_vec 