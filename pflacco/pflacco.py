from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import BoolVector, FloatVector, FloatMatrix, IntVector, StrVector
from rpy2.robjects import ListVector, NA_Complex, NA_Character, NA_Integer
import rpy2.robjects.packages as rpackages
import rpy2.robjects as R
from rpy2.robjects import numpy2ri
import numpy as np

base = importr('base')
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
packnames = ['flacco', 'dplyr', 'lhs', 'expm', 'mlbench', 'numDeriv', 'shape', 'shiny', 'testthat', 'RANN', 'mda']
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

flacco = importr('flacco')

def _translate_control(control):
    """
    Transforms a python dict to a valid R object
    Args:
      control: python dict

    Returns: R object of type ListVector

    """
    ctrl = {}
    for key, lst in control.items():
        if isinstance(lst, list):
            if all(isinstance(n, int) for n in lst):
                entry = IntVector(control[key])
            elif all(isinstance(n, bool) for n in lst):
                entry = BoolVector(control[key])
            elif all(isinstance(n, float) for n in lst):
                entry = FloatVector(control[key])
            elif all(isinstance(n, str) for n in lst):
                entry = StrVector(control[key])
            else:
                entry = None
            if entry is not None:
                ctrl[key] = entry
        else:
            ctrl[key] = lst
    return ListVector(ctrl)


def summary(r_object):
    """
    Interface to R's summary function.

    Args:
      r_object: R object on which R's summary() is invoked on
    
    """
    base.summary(r_object)

def printr(r_object):
    """
    Interface to R's print function
    Args:
      r_object: R object on which R's print() is invoked on

    """
    base.print(r_object)

def calculate_features(feat_object, control = {}):
    """
    Performs an Exploratory Landscape Analysis of a continuous function and computes various features, which quantify the function's landscape.
    Args:
      feat_object: R feature objected as created by create_feature_object()
      control:  python dict containing various settings for feature set. For more details see https://cran.r-project.org/web/packages/flacco/flacco.pdf. (Default value = {})

    Returns:

    """
    control = _translate_control(control)
    feat_set = flacco.calculateFeatures(feat_object, control)
    return {key : feat_set.rx2(key)[0] for key in feat_set.names}


def calculate_feature_set(feat_object, set_name, control = {}):
    """
    Performs an Exploratory Landscape Analysis of a continuous function and computes various features, which quantify the function's landscape for a single feature set.
    
    Args:
      feat_object: R feature objected as created by create_feature_object()
      set_name: feature set name, use list_all_available_feature_sets() to get an overview 
      control:  python dict containing various settings for feature set. For more details see https://cran.r-project.org/web/packages/flacco/flacco.pdf. (Default value = {})

    Returns:

    """
    control = _translate_control(control)
    feat_set = flacco.calculateFeatureSet(feat_object, set_name, control)
    return {key : feat_set.rx2(key)[0] for key in feat_set.names}

def create_initial_sample(n_obs, dim, type = 'lhs', lower_bound = None, upper_bound = None):
    """
    Convenient helper function, which creates an initial sample - either based on random (uniform) sampling or using latin hypercube sampling.

    Args:
      n_obs: number of observations
      dim: number of dimensions
      type: type of sampling strategy (Default value = 'lhs')
      lower_bound: The lower bounds of the initial sample as a list of size dim (Default value = 0)
      upper_bound: The upper bounds of the initial sample as a list of size dim (Default value = 1)

    Returns: numpy array of shape (n_obs x dim)

    """
    if lower_bound is None:
        lower_bound = [0] * dim
    if upper_bound is None:
        upper_bound = [1] * dim

    pcontrol = {
        'init_sample.type': type,
        'init_sample.lower': IntVector(lower_bound),
        'init_sample.upper': IntVector(upper_bound)}

    return np.array(flacco.createInitialSample(n_obs, dim, ListVector(pcontrol)))

# TODO: eventually one could add the objective function and pass is down to R.
def create_feature_object(x, y, minimize=True, lower = 0, upper = 1, blocks = None):
    """
    Creates a FeatureObject which will be used as input for all the feature computations.,

    Args:
      x: numpy 2D array containing the initial sample
      y: list containing the objective values of the initial sample 
      minimize: logical variable defining whether the objective is to minimize or not (Default value = True)
      lower:  python list or integer defining the lower limits per dimension (Default value = 0)
      upper:  python list or integer defining the lower limits per dimension (Default value = 1)
      blocks: number of blocks per dimension (Default value = None)

    Returns: rpy2.robject

    """
    numpy2ri.activate()
    x = R.r.matrix(x, nrow = len(x))
    numpy2ri.deactivate()
    y = FloatVector(y)

    if blocks is None:
        result = flacco.createFeatureObject(X = x, y = y, minimize = minimize, lower = lower, upper = upper, force = False)
    else:
        blocks = IntVector(blocks) if isinstance(blocks, list) else IntVector([blocks])
        result = flacco.createFeatureObject(X = x, y = y, minimize = minimize, lower = lower, upper = upper, blocks = blocks, force = False)

    return result

def list_available_feature_sets(allow_cellmapping = True, allow_additional_costs = True):
    """
    Lists all available feature sets w.r.t. certain restrictions.

    Args:
      allow_cellmapping: Determines whether cell maping features should be considered or not (Default value = True)
      allow_additional_costs: Determines whether feature sets which induce additional function evaluations should be considered or not (Default value = True)

    Returns:

    """
    feature_sets = flacco.listAvailableFeatureSets(subset = subset, allow_cellmapping = allow_cellmapping, allow_additional_costs = allow_additional_costs)
    print(feature_sets)
    return feature_sets

def plot_barrier_tree_2d():
    """ 
    Creates a 2D image containing the barrier tree of this cell mapping.

    """
    control = _translate_control(control)
    flacco.plotBarrierTree2D(feat_object, control)
    input("Press Enter to continue...")

def plot_barrier_tree_3d():
    """ 
    Creates a 3D surface plot containing the barrier tree of this cell mapping.

    """
    control = _translate_control(control)
    flacco.plotBarrierTree3D(feat_object, control)
    input("Press Enter to continue...")

def plot_cell_mapping(feat_object, control = {}):
    """
    Visualizes the transitions among the cells in the General Cell Mapping approach.

    Args:
      feat_object: R feature objected as created by create_feature_object()
      control: python dict containing various control arguments. For more details see the section about 'plotCellMapping' https://cran.r-project.org/web/packages/flacco/flacco.pdf. (Default value = {})

    """
    control = _translate_control(control)
    flacco.plotCellMapping(feat_object, control)
    input("Press Enter to continue...")

def plot_information_content(feat_object, control = {}):
    """
    Creates a plot of the Information Content Features.

    Args:
      feat_object: R feature objected as created by create_feature_object()
      control:  python dict containing various control arguments. For more details see the section about 'plotInformationContent' https://cran.r-project.org/web/packages/flacco/flacco.pdf. (Default value = {})
    Returns:

    """
    control = _translate_control(control)
    flacco.plotInformationContent(feat_object, control)
    input("Press Enter to continue...")

def run_flacco_gui():
    """ 
    Starts a shiny application, which allows the user to compute the flacco features and also visualize the underlying functions with an graphical user interface.

    """
    flacco.runFlaccoGUI()
