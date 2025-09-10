import matplotlib.pyplot as plt
from typing import Dict, List, Literal, Tuple

import networkx as nx
import torch
from tqdm import tqdm

from .client import FLClient
import numpy as np
import logging



Topology = Literal["ring", "erdos_renyi", "random"]

log = logging.getLogger(__name__)

def plot_topology(graph: nx.Graph, title: str = "Topology", path: str = "topology.png") -> None:
    """
    Plot the given networkx graph topology.
    """
    pos = nx.spring_layout(graph) if not nx.get_node_attributes(graph, 'pos') else nx.get_node_attributes(graph, 'pos')
    plt.figure(figsize=(6, 6))
    nx.draw_networkx(graph, pos, with_labels=False, node_color='skyblue', edge_color='gray', node_size=700)
    plt.title(title)
    plt.axis('off')
    plt.savefig(path)
    plt.close()

    
def build_topology(num_clients: int, cfg: Dict, seed: int = 42) -> nx.Graph:
    if cfg.topology == "ring":
        graph = nx.cycle_graph(num_clients)
    elif cfg.topology == "erdos_renyi":
        graph = nx.erdos_renyi_graph(num_clients, cfg.er_p, seed=seed)
    elif cfg.topology == "random":
        graph = nx.gnm_random_graph(num_clients, max(1, int(cfg.er_p * num_clients * (num_clients - 1) / 2)), seed=seed)
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
        metrics['train'].append(train_results)
        # Share with neighbors and aggregate
        neighbor_states: List[Dict[str, Dict[str, torch.Tensor]]] = []

        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            selected_neighbors = np.random.choice(neighbors, size=int(len(neighbors)*participation_rate))
            if not node in selected_neighbors:
                selected_neighbors = np.concatenate((selected_neighbors, [node]))
            total_samples_per_node = sum([len(clients[n].train_loader.dataset) for n in selected_neighbors])
            # We multiply each gradient by n_k/n
            gradients = [aggregate_gradients_weighted([clients[n].get_gradient()], [len(clients[n].train_loader.dataset)/total_samples_per_node])for n in selected_neighbors]
            
            weights = [graph.get_edge_data(node, n)["weight"]*len(neighbors)/len(selected_neighbors) for n in selected_neighbors]
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
            test_metrics = {"loss": avg_loss, "accuracy": avg_acc}
            metrics["test"].append(test_metrics)
    return metrics 