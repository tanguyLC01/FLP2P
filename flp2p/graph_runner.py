import matplotlib.pyplot as plt
from typing import Dict, List, Literal, Tuple

import networkx as nx
import torch
from tqdm import tqdm

from .client import FLClient
import random


Topology = Literal["ring", "erdos_renyi"]



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
    
def build_topology(num_clients: int, topology: Topology = "ring", er_p: float = 0.2, seed: int = 42) -> nx.Graph:
    if topology == "ring":
        return nx.cycle_graph(num_clients)
    if topology == "erdos_renyi":
        return nx.erdos_renyi_graph(num_clients, er_p, seed=seed)
    raise ValueError(f"Unknown topology: {topology}")


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
) -> List[Tuple[float, float]]:
    metrics: List[Tuple[float, float]] = []
    for rnd in tqdm(range(rounds), disable=not progress, desc="Rounds"):
        
        # Local training
        for client in clients:
            client.local_train(local_epochs=local_epochs)
            
        # Share with neighbors and aggregate
        neighbor_states: List[Dict[str, Dict[str, torch.Tensor]]] = []

        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            selected_neighbors = [n for n in neighbors if random.random() < 0.5]
            if not selected_neighbors:
                neighbor_states.append({})
                continue
            gradients = [clients[n].get_gradient() for n in selected_neighbors]
            weights = [graph.get_edge_data(node, n).get("weight", 1.0) for n in selected_neighbors]
            aggregated = aggregate_gradients_weighted(gradients, weights)
            # Store or use aggregated as needed, e.g., append to neighbor_states
            neighbor_states.append(aggregated)
    
        # Apply aggregated shared states
        for node, agg_state in enumerate(neighbor_states):
            clients[node].update_state(agg_state)
            
        # Evaluate (average across clients)
        with torch.no_grad():
            losses, accs = [], []
            for client in clients:
                loss, acc = client.evaluate()
                losses.append(loss)
                accs.append(acc)
        metrics.append((sum(losses) / len(losses), sum(accs) / len(accs)))
    print(metrics)
    return metrics 