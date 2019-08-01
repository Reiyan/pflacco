from rpy2.robjects.vectors import StrVector, FloatVector, FloatMatrix
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri
import rpy2.robjects as R
import numpy as np

from rpy2.robjects.packages import importr




base = importr('base')
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
packnames = ['flacco', 'dplyr', 'lhs']
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

flacco = importr('flacco')
summary = base.summary

 
    # def get_initial_sample(self, dim, sampling='random', lbound = None, ubound=None):
    #     if lbound is None:
    #         lbound = [0]*dim
    #     if ubound is None:
    #         ubound = [1]*dim
    #     n_obs = 50*dim
    #     lbound = [str(i) for i in lbound]
    #     ubound = [str(i) for i in ubound]
    #     r_control = 'list(init_sample.type = "' + str(sampling) + '", init_sample.lower = c(' + ",".join(lbound) + \
    #         '), init_sample.upper = c(' + ",".join(ubound) + "))"
    #     r_init_sample = 'createInitialSample(n.obs = ' + str(n_obs) + ', dim = ' + str(dim) + ', control = ' + r_control + ')'
    #     x = R.r(r_init_sample)
    #     sample = []
    #     for d in range(dim):
    #         sample.append(x[int((d*len(x)/dim)):int((((d+1)*len(x)/dim)))])
    #     return np.column_stack(sample)

def get_ela_features(x_values, y_values, dim):
    create_feature_object = R.r['create_feat_obj']
    calculate_feature_set = R.r['calc_feat_set']
    y = FloatVector(y_values)


    rpy2.robjects.numpy2ri.activate()
    nr, nc = x_values.shape
    X = R.r.matrix(x_values, nrow=nr, ncol=nc)

    feat_obj = create_feature_object(X,y)
    feat_set = ['ela_distr', 'ela_level', 'ela_meta', 'ic', 'nbc']
    feat_set_data = calculate_feature_set(feat_obj, StrVector(feat_set))
    
    return feat_set_data

def bbob_import():
    return None

def bbob_import_page():
    return None

def calculate_feature_set():
    return None

def compute_grid_centers():
    return None

def convert_init_design_to_grid():
    return None

def create_initial_sample(n_obs, dim, control={}):
    result =  flacco.createInitialSample(n_obs, dim, control)
    return result

def feature_list():
    return None

def feature_object():
    return None

def create_feature_object(init=None, x=None, y=None, fun=None, minimize=True, lower = -5, upper = 5, blocks = 3, objective = 'y', force = False):
    result = flacco.createFeatureObject(init, x, y, fun, minimize, lower, upper, blocks, objective, force)
    print('lul')

def feature_object_sidebar():
    return None

def feature_set_calculation():
    return None

def feature_set_calculation_component():
    return None

def feature_set_visualization():
    return None

def feature_set_visualization_component():
    return None

def find_linear_neighbours():
    return None

def find_nearest_prototype():
    return None

def function_input():
    return None

def list_available_feature_sets():
    return None

def measure_time():
    return None

def plot_barrier_tree_2d():
    return None

def plot_barrier_tree_3d():
    return None

def plot_cell_mapping():
    return None

def plot_feature_importance():
    return None

def plot_information_content():
    return None

def run_flacco_gui():
    return None

def smoof_import():
    return None

def smoof_import_page():
    return None
# import random
# t = FlaccoInterface()
# sample = t.get_initial_sample(2)
# y = []
# for i in range(len(sample)):
#     y.append(random.uniform(0, 1))

# ela = t.get_ela_features(sample, y, 2)
# print(ela)


ctrl = {
    'init_sample.type': 'lhs',
    'init_sample.lower': [-5, 2, 0],
    'init_sample.upper': 10
}
sample = create_initial_sample(100, 2, ctrl)
create_feature_object()