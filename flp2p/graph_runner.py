import matplotlib.pyplot as plt
from typing import Dict, List, Literal, Tuple

import networkx as nx
import torch
from tqdm import tqdm

from .client import FLClient
import numpy as np
import logging
from pyvis.network import Network



Topology = Literal["ring", "erdos_renyi", "random"]

log = logging.getLogger(__name__)

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

    
def build_topology(num_clients: int, cfg: Dict, seed: int = 42) -> nx.Graph:
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
    else:
        raise ValueError(f"Unknown topology: {cfg.topology}")

    # Ensure self-loops and set edge weights to 1/d for each node
    for node in graph.nodes():
        if not graph.has_edge(node, node):
            graph.add_edge(node, node)
    for node in graph.nodes():
        d = graph.degree[node]
        for neighbor in graph.neighbors(node):
            graph[node][neighbor]["weight"] = 1.0 / d
    return graph



def average_states(state_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not state_list:
        return {}
    avg: Dict[str, torch.Tensor] = {}
    for key in state_list[0].keys():
        tensors = [s[key].float() for s in state_list if key in s]
        if not tensors:
            continue
        stacked = torch.stack(tensors, dim=0)
        avg[key] = stacked.mean(dim=0)
    return avg

def aggregate_gradients_weighted(gradients_list: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    """
    Aggregate gradients weighted by the adjacency matrix (weights).
    Args:
        gradients_list: List of gradients dicts (param_name -> tensor) from neighbors.
        weights: List of weights (same order as gradients_list) from adjacency matrix.
    Returns:
        Dict of aggregated gradients (param_name -> tensor).
    """
    if not gradients_list or not weights or len(gradients_list) != len(weights):
        return {}
    agg: Dict[str, torch.Tensor] = {}
    param_names = gradients_list[0].keys()
    for name in param_names:
        weighted_grads = [g[name].float() * w for g, w in zip(gradients_list, weights) if name in g]
        if weighted_grads:
            agg[name] = sum(weighted_grads)
    return agg


def run_rounds(
    clients: List[FLClient],
    graph: nx.Graph,
    rounds: int = 5,
    local_epochs: int = 1,
    participation_rate: float = 0.5,
    progress: bool = True,
) -> Dict[str, List[Tuple[float, float]]]:
    metrics: Dict[str, List[Tuple[float, float]]] = {"train": [], 'test': []}
    for rnd in tqdm(range(rounds), disable=not progress, desc="Rounds"):
        
        # Local training
        correct, train_loss, train_samples, train_gradient_norm = 0, 0, 0, 0
        for client in clients:
            loss, acc, n_samples, gradient_norm = client.local_train(local_epochs=local_epochs)
            train_samples += n_samples
            correct  += acc * n_samples
            train_loss  += loss * n_samples
            train_gradient_norm += gradient_norm
        train_results = {'loss': train_loss/train_samples, 'accuracy': correct/train_samples, 'gradient_norm': train_gradient_norm/len(clients)}
        log.info(f"Train, Round {rnd} : loss => {train_loss/train_samples},  accuracy: {correct/train_samples}, gradient_norm : {train_gradient_norm/len(clients)}")
        metrics['train'].append(train_results)
        
        # Share with neighbors and aggregate
        neighbor_states: List[Dict[str, Dict[str, torch.Tensor]]] = []

        # For each edge, decide if it is active this round (bidirectional selection)
        edges = np.array(list(graph.edges()))
        idxs = np.random.choice(len(edges), int(participation_rate * len(edges)), replace=False)
        active_edges = edges[idxs]

        # For each node, collect its selected neighbors (including self)
        selected_neighbors_per_node = []
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            selected_neighbors = [n for n in neighbors if (node, n) or (n, node) in active_edges]
            if node not in selected_neighbors:
                selected_neighbors.append(node)
            selected_neighbors_per_node.append(selected_neighbors)
            
        for node, selected_neighbors in enumerate(selected_neighbors_per_node):
            total_samples_per_node = sum([len(clients[n].train_loader.dataset) for n in selected_neighbors])
            # We multiply each gradient by n_k/n
            gradients = [
            aggregate_gradients_weighted(
                [clients[n].get_gradient()],
                [len(clients[n].train_loader.dataset) / total_samples_per_node]
            )
            for n in selected_neighbors
            ]
            node_degree = graph.degree[node]
            weights = [
                graph.get_edge_data(node, n).get("width", graph.get_edge_data(n, node)["width"]) * node_degree / len(selected_neighbors)
                for n in selected_neighbors
            ]
            aggregated = aggregate_gradients_weighted(gradients, weights)
            neighbor_states.append(aggregated)
    
        # Apply aggregated shared states
        for node, agg_state in enumerate(neighbor_states):
            clients[node].update_state(agg_state)
            
        # Evaluate (average across clients)
        with torch.no_grad():
            total_samples = 0
            weighted_loss = 0.0
            weighted_acc = 0.0
            for client in clients:
                loss, acc = client.evaluate()
                num_samples = len(client.test_loader.dataset)
                weighted_loss += loss * num_samples
                weighted_acc += acc * num_samples
                total_samples += num_samples
            avg_loss = weighted_loss / total_samples
            avg_acc = weighted_acc / total_samples
            test_metrics = {"loss": avg_loss, "accuracy": avg_acc}
            log.info(f"Test, Round {rnd} : loss => {avg_loss},  accuracy: {avg_acc}")
            metrics["test"].append(test_metrics)
    return metrics 