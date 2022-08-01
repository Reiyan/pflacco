import pandas as pd
import scipy.optimize as opt
import numpy as np
import random
#import networkx as nx
#import os

from pflacco_utils import _transform_bounds_to_canonical

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

def compute_local_optima_network(f, dim, lower_bound, upper_bound, random_seed = None, logs=False, stepsize = 2, basin_hopping_iteration = 100, stopping_threshold = 1000, minimizer_kwargs = None):
    global nodes, edges, last, restart
    restart = True
    nodes = []
    edges = []
    last = 0

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

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


def compute_lon_features(nodes, edges, f_opt = None):
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
