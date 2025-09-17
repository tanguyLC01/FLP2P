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

    for node in graph.nodes():
        d = graph.degree[node] +  1  # +1 to include self-loop
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




def run_rounds(
    clients: List[FLClient],
    graph: nx.Graph,
    rounds: int = 5,
    local_epochs: int = 1,
    participation_rate: float = 0.5,
    progress: bool = True,
    consensus_lr: int = 0.1,
) -> Dict[str, List[Tuple[float, float]]]:
    metrics: Dict[str, List[Tuple[float, float]]] = {"train": [], 'test': []}
    
    for client in clients:
        client.total_neighbors_samples = sum(len(clients[n].train_loader.dataset) for n in graph.neighbors(clients.index(client))) 
        
    for rnd in tqdm(range(rounds), disable=not progress, desc="Rounds"):
        
        # For each edge, decide if it is active this round (bidirectional selection)
        edges = np.array(list(graph.edges()))
        idxs = np.random.choice(len(edges), int(participation_rate * len(edges)), replace=False)
        active_edges = edges[idxs]
        
        # Local training (only for clients with a vertex in active_edges)
        active_nodes = set(active_edges.flatten())
        correct, train_loss, train_samples, train_gradient_norm = 0, 0, 0, 0
        gradients_per_client = {}
        for idx, client in enumerate(clients):
            if idx in active_nodes:
                loss, acc, n_samples, gradient_norm, avg_gradients = client.local_train(local_epochs=local_epochs)
                train_samples += n_samples
                correct += acc * n_samples
                train_loss += loss * n_samples
                gradients_per_client[idx] = avg_gradients
                train_gradient_norm += gradient_norm
        train_results = {
        'loss': train_loss / train_samples,
        'accuracy': correct / train_samples,
        'gradient_norm': train_gradient_norm / max(1, len(active_nodes))
        }
        log.info(f"Train, Round {rnd} : loss => {train_results['loss']},  accuracy: {train_results['accuracy']}, gradient_norm : {train_results['gradient_norm']}")
        metrics['train'].append(train_results)
        
        
        for active_node in active_nodes:
            neighbors = list(graph.neighbors(active_node))
            neighbor_gradients = {n: gradients_per_client[n] for n in neighbors if n in gradients_per_client}
            clients[active_node].store_neighbor_gradients(neighbor_gradients)
        
        # # Share with neighbors and aggregate
        # neighbor_states: List[Dict[str, Dict[str, torch.Tensor]]] = []

        # # For each node, collect its selected neighbors (including self)
        # selected_neighbors_per_node = []
        # for node in graph.nodes:
        #     neighbors = list(graph.neighbors(node))
        #     selected_neighbors = [n for n in neighbors if (node, n) or (n, node) in active_edges]
        #     if node not in selected_neighbors:
        #         selected_neighbors.append(node)
        #     selected_neighbors_per_node.append(selected_neighbors)
            
        # # Each node aggregates the gradients of its selected neighbors and add the gradient of previous round 
        # for node, selected_neighbors in enumerate(selected_neighbors_per_node):
           
        #     # We multiply each gradient by n_k/n
        #     # gradients = [
        #     # aggregate_gradients_weighted(
        #     #     [clients[n].get_gradient()],
        #     #     [len(clients[n].train_loader.dataset) / total_samples_per_node]
        #     # )
        #     # for n in selected_neighbors
        #     # ]  
        #     gradients = [
        #         clients[n].get_gradient()
        #     for n in selected_neighbors
        #     ]
            
        #     node_degree = graph.degree[node]
        #     weights = [
        #         graph.get_edge_data(node, n).get("weight")  * node_degree / len(selected_neighbors)
        #         for n in selected_neighbors
        #     ]
        #     # weights = list(np.ones(len(selected_neighbors)))
        #     aggregated = aggregate_gradients_weighted(gradients, weights)
        #     neighbor_states.append(aggregated)
    
        # Apply aggregated shared states
        for node in active_nodes:
            weights = list(np.ones(len(graph.neighbors(node))))
            clients[node].update_state(weights, consensus_lr=consensus_lr)
            
        # Evaluate (average across clients)
        with torch.no_grad():
            total_samples = 0
            weighted_loss = 0.0
            weighted_acc = 0.0
            for node in active_nodes:
                client = clients[node]
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