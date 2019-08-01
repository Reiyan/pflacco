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
packnames = ['flacco', 'dplyr', 'lhs', 'expm', 'mlbench', 'numDeriv', 'shape', 'shiny', 'testthat']
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

flacco = importr('flacco')

def _translate_control(control):
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
    base.summary(r_object)

def printr(r_object):
    base.print(r_object)

def calculate_features(feat_object, control = {}):
    control = _translate_control(control)
    feat_set = flacco.calculateFeatures(feat_object, control)
    return {key : feat_set.rx2(key)[0] for key in feat_set.names}


def calculate_feature_set(feat_object, set_name, control = {}):
    control = _translate_control(control)
    feat_set = flacco.calculateFeatureSet(feat_object, set_name, control)
    return {key : feat_set.rx2(key)[0] for key in feat_set.names}

def compute_grid_centers():
    return None

def convert_init_design_to_grid():
    return None

def create_initial_sample(n_obs, dim, type = 'lhs', lower_bound = None, upper_bound = None):
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
def create_feature_object(x, y, minimize=True, lower = 0, upper = 1, blocks = None, force = False):

    numpy2ri.activate()
    x = R.r.matrix(x, nrow = len(x))
    numpy2ri.deactivate()
    y = FloatVector(y)

    if blocks is None:
        result = flacco.createFeatureObject(X = x, y = y, minimize = minimize, lower = lower, upper = upper, force = force)
    else:
        blocks = IntVector(blocks) if isinstance(blocks, list) else IntVector([blocks])
        result = flacco.createFeatureObject(X = x, y = y, minimize = minimize, lower = lower, upper = upper, blocks = blocks, force = force)

    return result

def list_available_feature_sets(allow_cellmapping = True, allow_additional_costs = True):
    feature_sets = flacco.listAvailableFeatureSets(subset = subset, allow_cellmapping = allow_cellmapping, allow_additional_costs = allow_additional_costs)
    print(feature_sets)
    return feature_sets

def plot_barrier_tree_2d():
    control = _translate_control(control)
    flacco.plotBarrierTree2D(feat_object, control)
    input("Press Enter to continue...")

def plot_barrier_tree_3d():
    control = _translate_control(control)
    flacco.plotBarrierTree3D(feat_object, control)
    input("Press Enter to continue...")

def plot_cell_mapping(feat_object, control = {}):
    control = _translate_control(control)
    flacco.plotCellMapping(feat_object, control)
    input("Press Enter to continue...")

def plot_information_content(feat_object, control = {}):
    control = _translate_control(control)
    flacco.plotInformationContent(feat_object, control)
    input("Press Enter to continue...")

def run_flacco_gui():
    flacco.runFlaccoGUI()
