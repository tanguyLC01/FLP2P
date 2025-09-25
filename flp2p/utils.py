from pyvis.network import Network
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Dict
from .graph_runner import GOSSIPING

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
    html_path = path if path.endswith(".html") else path + ".html"
    net.save_graph(html_path)

    
def build_topology(num_clients: int, cfg: Dict, mixing_matrix: GOSSIPING,seed: int = 42, consensus_lr: int = 0.1) -> nx.Graph:
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

        # Connect each node in cluster 2 to center2 (except center2 itself)
        for node in cluster2_nodes:
            if node != center2:
                graph.add_edge(center2, node)

        # Connect the two centers
        graph.add_edge(center1, center2)
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