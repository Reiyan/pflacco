import numpy as np
import pandas as pd
import random
import scipy.optimize as opt

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .utils import _transform_bounds_to_canonical

def _consolidate_edges(edges):
    edges = edges.groupby(['source', 'target']).size().reset_index(name='weight')
    return edges

def _consolide_nodes_same_fitness(nodes, edges, precision = 1e-5):
    result_nodes = nodes.copy()
    consolidated_nodes = {}
    for idx, node in nodes.iterrows():
        if idx in result_nodes.index:
            adjacent_nodes = nodes.iloc[edges[edges['source'] == idx]['target']]
            adjacent_nodes = adjacent_nodes[np.abs(adjacent_nodes['fval'] - node['fval']) < precision]
            if adjacent_nodes.shape[0] > 0:
                target_node_id = adjacent_nodes['id'].iloc[0]
                # check for reciprocal relationship between two nodes with same fval
                if str(target_node_id) not in consolidated_nodes.keys() and idx != target_node_id:
                    # Delete current node from results
                    result_nodes = result_nodes[result_nodes['id'] != idx]
                    # Update neutrality value
                    result_nodes.loc[target_node_id]['neutrality'] += node['neutrality']
                    # Delete edge between node and target_node with same fval
                    edges = edges[~((edges['source'] == idx) & (edges['target'] == target_node_id))]
                    # Transfer incoming edges of node to target_node
                    edges.loc[edges['target'] == idx, 'target'] = target_node_id
                    # transfer outgoing edges of node to target_node
                    edges.loc[edges['source'] == idx, 'source'] = target_node_id
                    consolidated_nodes[str(idx)] = target_node_id

    # Remove edges from nodes which point to itself (introduced by the consolidation of neutral nodes)
    edges = edges[~(edges['source'] == edges['target'])]

    return result_nodes, edges

# checks whether x is in the neighbourhood of any node
def _neighborhood(coord, threshold = 1e-5):
    node_index = -1
    duplicate = False
    # do not check anything if the list is empty
    if len(nodes) > 0: 
        # compute all differences in all dimension to given coord
        dist = [[abs(x1 - coord1) for (x1, coord1) in zip(x[1], coord)] for x in nodes]
        for i in range(len(dist)):
            if all(xi < threshold for xi in dist[i]):
                duplicate = True
                node_index = i
                break

    return duplicate, node_index

def _minfound(x, f, accept):
    global restart, last, log

    if accept:
        #Check if an existing local minimum is too close
        duplicate, node_index = _neighborhood(x)
        if not duplicate:
            #The AC only needs to be checked if it is not a restart
            nodes.append([f, x])
            if not restart:
                edges.append([last, len(nodes)-1])
            last = len(nodes)-1
        else:
            if not restart and not last == node_index:
                edges.append([last, node_index])
            last = node_index

        restart = False

'''
def create_graph(nodes, edges):
    # Create Graph
    graph = nx.DiGraph()
    for n in range(len(nodes)):
        graph.add_node(n, fval = nodes[n][0], x = nodes[n][1])
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    if not os.path.exists("lon_results"):
        os.mkdir("lon_results")
    #nx.drawing.nx_pydot.write_dot(graph, f'lon_results/dot_results/graph_{problem}_{fun_id}_{inst}_{dim}.dot')
'''

def compute_local_optima_network(
    f: Callable[[List[float]], float],
    dim: int,
    lower_bound: Union[List[float], float],
    upper_bound: Union[List[float], float],
    seed: Optional[int] = None,
    stepsize: int = 2,
    basin_hopping_iteration: int = 100,
    stopping_threshold: int = 1000,
    minimizer_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computation of a Local Optima Network in accordance to [1].

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
    seed : Optional[int], optional
        Seed for reproducability, by default None.
    stepsize : int, optional
        Maximum step size for use in the random displacement, by default 2
    basin_hopping_iteration : int, optional
        Number of independet basin-hopping runs. Each basin-hopping run consists of
        multiple iterations of a local search method, by default 100.
    stopping_threshold : int, optional
        Number of basin-hopping iterations. There will be a total of 
        ``basin_hopping_iteration + 1`` runs of the local minimizer, by default 1000
    minimizer_kwargs : Optional[Dict[str, Any]], optional
        Extra keyword arguments to be passed to the local minimizer
        ``scipy.optimize.minimize``, by default None.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple consisting of two Pandas dataframes. The first dataframe
        contains information about the nodes, i.e., the local optima.
        The second dataframe contains information about the edges.
    
    References
    ----------
    [1] Adair, J., Ochoa, G. and Malan, K.M., 2019, July.
        Local optima networks for continuous fitness landscapes.
        In Proceedings of the Genetic and Evolutionary Computation Conference Companion
        (pp. 1407-1414).
    """    
    global nodes, edges, last, restart
    restart = True
    nodes = []
    edges = []
    last = 0

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if minimizer_kwargs is None:
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'options': {
                'maxiter': 150,
                'ftol': 1e-7
            }
        }
    lower_bound, upper_bound = _transform_bounds_to_canonical(dim, lower_bound, upper_bound)
    minimizer_kwargs['bounds'] = list(zip(lower_bound, upper_bound))
    
    for _ in range(basin_hopping_iteration):
        x0 = np.random.uniform(lower_bound, upper_bound, dim)
        opt.basinhopping(f, x0, T=0.0, minimizer_kwargs=minimizer_kwargs, stepsize = stepsize, callback=_minfound, niter=stopping_threshold)
        restart = True

    nodes = pd.DataFrame(np.array([np.array([x, nodes[x][0], 1]) for x in range(len(nodes))]), columns = ['id', 'fval', 'neutrality'])
    edges = pd.DataFrame(edges, columns = ['source', 'target'])
    nodes, edges = _consolide_nodes_same_fitness(nodes, edges)
    edges = _consolidate_edges(edges)

    return nodes, edges

def calculate_lon_features(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    f_opt: float = None) -> Dict[str, Optional[Union[int, float]]]:
    """Calculation of Local Optima Network (LON) features based on a LON.

    Parameters
    ----------
    nodes : pd.DataFrame
        A dataframe containing all nodes of the LON.
        Can be created via the function ``compute_local_optima_network``.
    edges : pd.DataFrame
        A dataframe containg all edges of the LON.
        Can be created via the function ``compute_local_optima_network``.
    f_opt : float, optional
        Objective value of the global optimum (if known) of the objective
        function the LON was computed on, by default None.

    Returns
    -------
    Dict[str, Optional[Union[int, float]]]
        Dictionary consisting of the calculated features.
    """

    n_optima = len(nodes)
    neutral = nodes.loc[nodes['neutrality'] > 1, 'neutrality'].sum()/nodes['neutrality'].sum()
    n_funnels = len([x for x in nodes['id'] if x not in edges['source'].unique().tolist()])

    result = {
        'lon.n_optima': n_optima,
        'lon.neutral_nodes_proportion': neutral,
        'lon.n_funnels': n_funnels,
        'lon.global_funnel_strength_norm': None
    }
    
    if f_opt is not None:
        gfunnels = nodes[round(nodes['fval'], 8) == round(f_opt, 8)]
        tmp = []
        for _, funnel_ in gfunnels.iterrows():
            loop_list = [funnel_['id']]
            while len(loop_list) > 0:
                node = loop_list.pop()
                tmp.append(node)
                inc_edges = edges.loc[edges['target'] == node, 'source']
                loop_list.extend(inc_edges.tolist())

        result['lon.global_funnel_strength_norm'] = len(np.unique(tmp))/n_optima

    return result
