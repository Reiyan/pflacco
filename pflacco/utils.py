import itertools
import warnings

import numpy as np
import pandas as pd

# from rpy2.robjects.packages import importr, isinstalled

# def _interface_mda():
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
        raise Exception("Unknown format of X. X must be either a Python list, numpy array oder pandas DataFrame")

    X.columns = ["x" + str(x) for x in range(X.shape[1])]
    X = X.reset_index(drop=True)
    if not isinstance(y, pd.Series) and (isinstance(y, np.ndarray) or isinstance(y, list)):
        y = pd.Series(y, name="y")
    y = y.reset_index(drop=True)

    if X.shape[0] != len(y):
        raise Exception("pd.DataFrame X and pd.Series y must provide the same amount of observation.")
    return X, y


# Min-max normalization of the objective values, cf. Prager and Trautmann (2023):
# "Nullifying the Inherent Bias of Non-invariant Exploratory Landscape Analysis Features".
def _normalize_objective(y):
    y_norm, _, _ = _normalize_objective_with_scale(y)
    return y_norm


# Same normalization, but additionally returns the offset and range it was based on.
# Feature sets which evaluate the objective function themselves need those to wrap `f`
# in the very same transformation, e.g. f_norm = lambda x: (f(x) - y_min)/y_range.
def _normalize_objective_with_scale(y):
    y_min = y.min()
    y_range = y.max() - y_min
    if y_range == 0:
        # A constant objective offers no range to normalize against. Report this the
        # way the feature computations report degenerate input, i.e. with NaN.
        warnings.warn(
            "The objective values are constant and can therefore not be normalized. All features which depend on the objective values will be NaN."
        )
        return y * np.nan, y_min, np.nan
    return (y - y_min) / y_range, y_min, y_range


# Helper function to transform scalar bounds to an N * D array, where D is the dimensionality and N the different lower/upper bounds of the respective dimensions.
def _transform_bounds_to_canonical(dim, lower_bound, upper_bound):
    if lower_bound is None or upper_bound is None:
        raise Exception(
            "No values for either lower bounds or upper bounds where provided. Pflacco is only applicable to box constrainted problems."
        )
    if not isinstance(lower_bound, list) and type(lower_bound) is not np.ndarray:
        lower_bound = np.array([lower_bound] * dim)
    if isinstance(lower_bound, list):
        lower_bound = np.array(lower_bound)
    if not isinstance(upper_bound, list) and type(upper_bound) is not np.ndarray:
        upper_bound = np.array([upper_bound] * dim)
    if isinstance(upper_bound, list):
        upper_bound = np.array(upper_bound)
    if len(lower_bound) != dim or len(upper_bound) != dim:
        raise Exception("Length of lower-/upperbound is not the same as the problem dimension")
    if not (lower_bound < upper_bound).all():
        raise Exception("Not all elements of lower bound are smaller than upper bound")

    return lower_bound, upper_bound


def _determine_max_n_blocks(X):
    # The grid spans blocks^d cells and every cell should hold at least three
    # observations, i.e. the sample size n has to satisfy 3 * blocks^d <= n.
    blocks = 1
    while 3 * ((blocks + 1) ** X.shape[1]) <= X.shape[0]:
        blocks += 1
    return blocks


def _check_blocks_variable(X, dim, blocks, force=False):
    # A block count that pflacco picked itself is reported as a warning rather than
    # rejected, since the caller did not provide a value which could be lowered.
    auto = blocks is None
    if auto:
        blocks = max(_determine_max_n_blocks(X), 2)

    if not isinstance(blocks, list) and type(blocks) is not np.ndarray:
        blocks = np.array([blocks] * dim)
    elif isinstance(blocks, list):
        blocks = np.array(blocks)
    if len(blocks) != dim:
        raise Exception('The provided value for "block" does not have the same length as the dimensionality of X.')

    # Minimum sample size to retain three observations per cell.
    min_sample_size = 3 * np.prod(blocks)
    if min_sample_size > X.shape[0]:
        if force or auto:
            warnings.warn(
                f"For the given dataframe X, blocks {blocks.tolist()} require at least {min_sample_size} observations to retain 3 observations per cell. X only provides {X.shape[0]}."
            )
        else:
            raise ValueError(
                'The provided value for "block" is too large, resulting in less than 3 observations per cell.'
            )

    return blocks


def _create_blocks(X, y, lower_bound, upper_bound, blocks=None):
    X, y = _validate_variable_types(X, y)
    dim = X.shape[1]
    lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)

    block_widths = (upper_bound - lower_bound) / blocks
    cp = np.cumprod(np.insert(blocks, 0, 1))

    cell_ids = []
    for idx, row in X.iterrows():
        cid = [
            cp[ndim] * np.floor((row.iloc[ndim] - lower_bound[ndim]) / block_widths[ndim]) for ndim in range(X.shape[1])
        ]
        cell_ids.append((cid - cp[:-1] * (row == upper_bound)).sum())
    cell_ids = np.array(cell_ids)

    n_centers = []
    for idx in range(len(blocks)):
        tmp = np.linspace(lower_bound[idx], upper_bound[idx], blocks[idx] + 1)
        n_centers.append((tmp[1:] + tmp[:-1]) / 2)
    # Artificial complicated sorting to replicate output from expand.grid in R
    c_centers = pd.DataFrame(np.array([x for x in itertools.product(*[y for y in n_centers])]))
    c_centers = c_centers.sort_values(list(reversed(c_centers.columns))).to_numpy()

    return cell_ids, c_centers
