import collections
from typing import List
import networkx as nx
import random
import pickle
import cvxpy as cp
import numpy as np

def decomposition(graph, size) -> list:
    node_degree = [[i, 0] for i in range(size)]
    node_to_node = [[] for i in range(size)]
    node_degree_dict = collections.defaultdict(int)
    node_set = set()
    for edge in graph:
        node1, node2 = edge[0], edge[1]
        node_degree[node1][1] += 1
        node_degree[node2][1] += 1
        if node1 in node_to_node[node2] or node2 in node_to_node[node1]:
            print("Invalid input graph! Double edge! ("+str(node1) +", "+ str(node2)+")")
            exit()
        if node1 == node2:
            print("Invalid input graph! Circle! ("+str(node1) +", "+ str(node2)+")")
            exit()

        node_to_node[node1].append(node2)
        node_to_node[node2].append(node1)
        node_degree_dict[node1] += 1
        node_degree_dict[node2] += 1
        node_set.add(node1)
        node_set.add(node2)

    node_degree = sorted(node_degree, key = lambda x: x[1])
    node_degree[:] = node_degree[::-1]
    subgraphs = []
    min_num = node_degree[0][1]
    while node_set:
        subgraph = []
        for i in range(size):
            node1, node1_degree = node_degree[i]
            if node1 not in node_set:
                continue
            for j in range(i+1, size):
                node2, node2_degree = node_degree[j]
                if node2 in node_set and node2 in node_to_node[node1]:
                    subgraph.append((node1, node2))
                    node_degree[j][1] -= 1
                    node_degree[i][1] -= 1
                    node_degree_dict[node1] -= 1
                    node_degree_dict[node2] -= 1
                    node_to_node[node1].remove(node2)
                    node_to_node[node2].remove(node1)
                    node_set.remove(node1)
                    node_set.remove(node2)
                    break
        subgraphs.append(subgraph)
        for node in node_degree_dict:
            if node_degree_dict[node] > 0:
                node_set.add(node)
        node_degree = sorted(node_degree, key = lambda x: x[1])
        node_degree[:] = node_degree[::-1]
    return subgraphs

def getSubGraphs(graph, size) -> list:
    subgraphs = list()
    M1 = nx.Graph()
    for i in range(len(graph.nodes)-1):
        M1 = nx.max_weight_matching(graph)
    if nx.is_perfect_matching(graph, M1):
        graph.remove_edges_from(list(M1))
        subgraphs.append(list(M1))
    else:
        edge_list = list(graph.edges)
        random.shuffle(edge_list)
        graph.remove_edges_from(edge_list)
        graph.add_edges_from(edge_list)

    # use greedy algorithm to decomposes the remaining part
    rpart = decomposition(list(graph.edges), size)
    for sgraph in rpart:
        subgraphs.append(sgraph)
    return subgraphs

        
def graphToLaplacian(subGraphs, size) -> List[np.ndarray]:
    L_matrices = list()
    for i, subgraph in enumerate(subGraphs):
        tmp_G = nx.Graph()
        tmp_G.add_edges_from(subgraph)
        tmp_G.add_nodes_from(range(size)) 
        L_matrices.append(nx.laplacian_matrix(tmp_G, nodelist=list(range(size))).todense())

    return L_matrices
    
def getProbability(L_matrices, commBudget) -> np.ndarray:
    num_subgraphs = len(L_matrices)
    p = cp.Variable(num_subgraphs)
    L = p[0]*L_matrices[0]
    for i in range(num_subgraphs-1):
        L += p[i+1]*L_matrices[i+1]
    eig = cp.lambda_sum_smallest(L, 2)
    sum_p = p[0]
    for i in range(num_subgraphs-1):
        sum_p += p[i+1]

    # cvx optimization for activation probabilities
    obj_fn = eig
    constraint = [sum_p <= num_subgraphs*commBudget, p>=0, p<=1]
    problem = cp.Problem(cp.Maximize(obj_fn), constraint)
    problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)

    # get solution
    tmp_p = p.value
    if tmp_p is None:
        print("CVX optimization failed!")
        exit()
    originActivationRatio = np.zeros((num_subgraphs))
    for i, pval in enumerate(tmp_p):
        originActivationRatio[i] = np.real(float(pval))

    return np.minimum(originActivationRatio, 1) 

def getAlpha(L_matrices, probabilities, num_nodes) -> float:
    num_subgraphs = len(L_matrices)
    
    # prepare matrices
    I = np.eye(num_nodes)
    J = np.ones((num_nodes, num_nodes))/num_nodes

    mean_L = np.zeros((num_nodes,num_nodes))
    var_L = np.zeros((num_nodes,num_nodes))
    for i in range(num_subgraphs):
        val = probabilities[i]
        mean_L += L_matrices[i]*val
        var_L += L_matrices[i]*(1-val)*val

    # SDP for mixing weight
    a = cp.Variable()
    b = cp.Variable()
    s = cp.Variable()
    obj_fn = s
    constraint = [(1-s)*I - 2*a*mean_L-J + b*(np.dot(mean_L,mean_L)+2*var_L) << 0, a>=0, s>=0, b>=0, cp.square(a) <= b]
    problem = cp.Problem(cp.Minimize(obj_fn), constraint)
    problem.solve(solver='CVXOPT', kktsolver=cp.ROBUST_KKTSOLVER)
    if a.value is None:
        print("CVX optimization failed!")
        exit()
    return  float(a.value)
