import matplotlib.pyplot as plt
from typing import Dict, List, Literal, Tuple

import networkx as nx
import torch
from tqdm import tqdm

from .client import FLClient
import random
import logging



Topology = Literal["ring", "erdos_renyi", "random"]

log = logging.getLogger(__name__)

def plot_topology(graph: nx.Graph, title: str = "Topology", path: str = "topology.png") -> None:
    """
    Plot the given networkx graph topology.
    """
    pos = nx.spring_layout(graph) if not nx.get_node_attributes(graph, 'pos') else nx.get_node_attributes(graph, 'pos')
    plt.figure(figsize=(6, 6))
    nx.draw_networkx(graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.axis('off')
    plt.savefig(path)
    plt.close()

    
def build_topology(num_clients: int, cfg: Dict, seed: int = 42) -> nx.Graph:
    if cfg.topology == "ring":
        return nx.cycle_graph(num_clients)
    elif topology == "erdos_renyi":
        return nx.erdos_renyi_graph(num_clients, er_p, seed=seed)
    elif topology == "random":
        topology = nx.gnm_random_graph(num_clients, max(1, int(er_p * num_clients * (num_clients - 1) / 2)), seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Ensure self-loops and set edge weights to 1/d for each node
    for node in topology.nodes():
        if not topology.has_edge(node, node):
            topology.add_edge(node, node)
    for node in topology.nodes():
        d = topology.degree[node]
        for neighbor in topology.neighbors(node):
            topology[node][neighbor]["weight"] = 1.0 / d
    return topology



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
    progress: bool = True,
) -> Dict[str, List[Tuple[float, float]]]:
    metrics: Dict[str, List[Tuple[float, float]]] = {"train": [], 'test': []}
    for rnd in tqdm(range(rounds), disable=not progress, desc="Rounds"):
        
        # Local training
        train_acc, train_loss = 0, 0
        train_samples = 0
        for client in clients:
            loss, acc, n_samples = client.local_train(local_epochs=local_epochs)
            train_samples += n_samples
            train_acc  += acc * n_samples
            train_loss  += loss * n_samples
        metrics['train'].append((train_loss/n_samples, train_acc/n_samples))
        # Share with neighbors and aggregate
        neighbor_states: List[Dict[str, Dict[str, torch.Tensor]]] = []

        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            selected_neighbors = [n for n in neighbors if random.random() < 0.5] + [node]
            gradients = [clients[n].get_gradient() for n in selected_neighbors]
            weights = [graph.get_edge_data(node, n).get("weight", 1/len(selected_neighbors)) for n in selected_neighbors]
            aggregated = aggregate_gradients_weighted(gradients, weights)
            # Store or use aggregated as needed, e.g., append to neighbor_states
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
            metrics["test"].append((avg_loss, avg_acc))
    return metrics 