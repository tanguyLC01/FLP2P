from pyvis.network import Network
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Dict, Literal
import torch

GOSSIPING = Literal['metropolis_hasting', 'maximum_degree', 'average', 'probability', 'matcha', 'jaccard']

def get_spectral_gap(matrix: np.array) -> float:
    eigenvals = set(np.abs(np.linalg.eigvals(matrix)))
    if len(eigenvals) == 1:
        return 1
    lambda_2 = sorted(eigenvals, reverse=False)[1]
    return 1 - lambda_2


def plot_topology(graph: nx.Graph, title: str = "Topology", path: str = "topology") -> None:
    """
    Plot the given networkx graph topology using PyVis and save a PNG image of it.
    """
    # Remove self-loops for visualization
    graph_no_self_loops = graph.copy()
    self_loops = list(nx.selfloop_edges(graph_no_self_loops))
    graph_no_self_loops.remove_edges_from(self_loops)

    net = Network(notebook=False, width="700px", height="700px", bgcolor="#222222", font_color="white")
    net.from_nx(graph_no_self_loops)
    for u, v, data in graph_no_self_loops.edges(data=True):
        if "probability_selection" in data:
            for edge in net.edges:
                if edge["from"] == u and edge["to"] == v:
                    edge["label"] = str(data["probability_selection"])
                    break
            
    html_path = path if path.endswith(".html") else path + ".html"
    net.save_graph(html_path)

    
def build_topology(num_clients: int, cfg: Dict, mixing_matrix: GOSSIPING,seed: int = 42) -> nx.Graph:
    if cfg.topology == "ring":
        graph = nx.cycle_graph(num_clients)
    elif cfg.topology == "erdos_renyi":
        graph = nx.erdos_renyi_graph(num_clients, cfg.er_p, seed=seed)
    elif cfg.topology == "random":
        graph = nx.gnm_random_graph(num_clients, max(1, int(cfg.er_p * num_clients * (num_clients - 1) / 2)), seed=seed)
    elif cfg.topology == "two_clusters":
        # Create two clusters, each with its own center node
        num_cluster1 = num_clients // 2

        # Assign node indices
        center1 = 0
        center2 = num_cluster1
        cluster1_nodes = list(range(0, num_cluster1))
        cluster2_nodes = list(range(num_cluster1, num_clients))

        graph = nx.Graph()
        graph.add_nodes_from(range(num_clients))

        # Connect each node in cluster 1 to center1 (except center1 itself)
        for node in cluster1_nodes:
            if node != center1:
                graph.add_edge(center1, node)
                graph[center1][node]["probability_selection"] = cfg.border_link_activation

        # Connect each node in cluster 2 to center2 (except center2 itself)
        for node in cluster2_nodes:
            if node != center2:
                graph.add_edge(center2, node)
                graph[center2][node]["probability_selection"] = cfg.border_link_activation

        # Connect the two centers
        graph.add_edge(center1, center2)
        graph[center1][center2]["probability_selection"] = cfg.main_link_activation
        
    elif cfg.topology == 'random_geometric':
        graph = nx.random_geometric_graph(num_clients, radius=cfg.radius, seed=seed)
        
    elif cfg.topology == "specific":
        edges = [edge for sublist in cfg.graph for edge in sublist]
        print(edges)
        # create a graph in NetworkX
        graph = nx.Graph()
        graph.add_edges_from(edges)
        
    else:
        raise ValueError(f"Unknown topology: {cfg.topology}")

    graph.remove_edges_from(nx.selfloop_edges(graph))
    if mixing_matrix == 'maximum_degree':
        max_degree = max([val for (_, val) in graph.degree()])
        for node in graph.nodes():
            for neighbor in graph.neighbors(node):
                graph[node][neighbor]["weight"] = 1/max_degree
    
    elif mixing_matrix == 'metropolis_hasting':
        for node in graph.nodes():
            for neighbor in graph.neighbors(node):
                graph[node][neighbor]["weight"] = 1/(1+max(graph.degree[node], graph.degree[neighbor]))

    return graph


def compute_consensus_distance(node_params):
    """Check how far apart node parameters are"""
    param_vectors = []
    for params in node_params:
        # Flatten all parameters into one vector
        flat_params = torch.cat([p.flatten() for p in params.values()])
        param_vectors.append(flat_params)
    
    # Compute pairwise distances
    distances = []
    for i in range(len(param_vectors)):
        for j in range(i+1, len(param_vectors)):
            dist = torch.norm(param_vectors[i] - param_vectors[j])
            distances.append(dist.item())
    
    return np.mean(distances), np.std(distances)

    
def compute_weight_matrix(graph, mixing_matrix: GOSSIPING ='metropolis_hasting'):
    """
    Compute the full N x N weight matrix W for all nodes.
    W[i,j] = weight from node i to node j
    """
    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    
    W = np.zeros((n_nodes, n_nodes))
    
    if mixing_matrix == 'metropolis_hasting' or mixing_matrix == 'probability':
        # First pass: compute edge weights
        for node in nodes:
            sum_neighbor_weights = 0
            
            for neighbor in graph.neighbors(node):
                # Metropolis-Hastings weight
                weight = 1 / (1 + max(graph.degree[node], graph.degree[neighbor]))
                W[node, neighbor] = weight
                sum_neighbor_weights += weight
            
            # Self-weight (diagonal)
            W[node, node] = 1 - sum_neighbor_weights
            
        if mixing_matrix == 'probability':  
            for u, v, data in graph.edges(data=True):
                p = data.get('probability_selection', 1.0)
                W[u, v] *= 1/p
                W[v, u] *= 1/p
                for i in range(W.shape[0]):
                    off_diag_sum = np.sum(W[i, :])
                    W[i, i] = 1 - off_diag_sum
                
    elif mixing_matrix == 'maximum_degree':
        max_degree = max(dict(graph.degree()).values())
        
        for node in nodes:
            
            # Edge weights
            for neighbor in graph.neighbors(node):
                W[node, neighbor] = 1 / max_degree
            
            # Self-weight
            W[node, node] = 1 - graph.degree[node] / max_degree
            
    elif mixing_matrix == 'jaccard':
        N = len(nodes)
        W = np.zeros((N, N))

        for i in nodes:
            neighbors_i = set(graph.neighbors(i))
            set_i = neighbors_i | {i}
            for j in neighbors_i:
                set_j = set(graph.neighbors(j)) | {j}
                # Jaccard Intersection over Union
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                W[i, j] = 1 - intersection / union

        # # 3. Add Self-Loops to make row sums exactly 1
        for i in range(N):
            W[i, i] = max(1.0 - np.sum(W[i, :]), 0)
            W[i,:] /= np.sum(W[i, :], axis=0)
    return W


def validate_weight_matrix(W):
    """Check if W is doubly stochastic"""
    n_nodes = W.shape[0]
    
    # Check row sums (should be 1.0)
    row_sums = np.sum(W, axis=1)
    print(f"Row sums - Min: {row_sums.min():.6f}, Max: {row_sums.max():.6f}")
    print(f"All rows sum to 1: {np.allclose(row_sums, 1.0)}")
    
    # Check non-negativity
    print(f"All weights non-negative: {np.all(W >= 0)}")
    
    #Check Symmetry
    print(f'W is symmetric: {np.all(W == W.T)}')
    # Check sparsity
    non_zero = np.count_nonzero(W)
    total = W.size
    print(f"Sparsity: {non_zero}/{total} ({100*non_zero/total:.1f}% non-zero)")
