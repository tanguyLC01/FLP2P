import matplotlib.pyplot as plt
from typing import Dict, List, Literal, Tuple

import networkx as nx
import torch
from tqdm import tqdm

from .client import FLClient
import numpy as np
import logging

log = logging.getLogger(__name__)

GOSSIPING = Literal['metropolis_hasting', 'maximum_degree']

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

def verify_data_split(client):
    # Sample a few examples from train and test
    train_batch = next(iter(client.train_loader))
    test_batch = next(iter(client.test_loader))
    
    print(f"Train batch shape: {train_batch[0].shape}")
    print(f"Test batch shape: {test_batch[0].shape}")
    
    # Check if there's overlap (shouldn't be any)
    train_hash = hash(train_batch[0].flatten().sum().item())
    test_hash = hash(test_batch[0].flatten().sum().item())
    print(f"Different data: {train_hash != test_hash}")
    
def compute_weight_matrix(graph, mixing_matrix: GOSSIPING ='metropolis_hasting'):
    """
    Compute the full N x N weight matrix W for all nodes.
    W[i,j] = weight from node i to node j
    """
    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    W = np.zeros((n_nodes, n_nodes))
    
    if mixing_matrix == 'metropolis_hasting':
        # First pass: compute edge weights
        for node in nodes:
            i = node_to_idx[node]
            sum_neighbor_weights = 0
            
            for neighbor in graph.neighbors(node):
                j = node_to_idx[neighbor]
                # Metropolis-Hastings weight
                weight = 1 / (1 + max(graph.degree[node], graph.degree[neighbor]))
                W[i, j] = weight
                sum_neighbor_weights += weight
            
            # Self-weight (diagonal)
            W[i, i] = 1 - sum_neighbor_weights
            
    elif mixing_matrix == 'maximum_degree':
        max_degree = max(dict(graph.degree()).values())
        
        for node in nodes:
            i = node_to_idx[node]
            
            # Edge weights
            for neighbor in graph.neighbors(node):
                j = node_to_idx[neighbor]
                W[i, j] = 1 / max_degree
            
            # Self-weight
            W[i, i] = 1 - graph.degree[node] / max_degree
    
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
    
    # Check sparsity
    non_zero = np.count_nonzero(W)
    total = W.size
    print(f"Sparsity: {non_zero}/{total} ({100*non_zero/total:.1f}% non-zero)")

def run_rounds(
    clients: List[FLClient],
    graph: nx.Graph,
    mixing_matrix: GOSSIPING,
    rounds: int = 5,
    local_epochs: int = 1,
    participation_rate: float = 0.5,
    progress: bool = True,
    consensus_lr: int = 0.1,
    lr_decay: int = 0,
    old_gradients: bool = True,
) -> Dict[str, List[Tuple[float, float]]]:
    metrics: Dict[str, List[Tuple[float, float]]] = {"train": [], 'test': []}

    metrics = {"train":
                                {"loss": [], "accuracy": []},
                    "test":
                                {"loss": [], "accuracy": []},
            }
    
    log.info("Data split check on random client")
    verify_data_split(np.random.choice(clients, 1)[0])
    
    # Compute the mixing matrix
    W = compute_weight_matrix(graph, mixing_matrix)
    validate_weight_matrix(W)
    
    for rnd in tqdm(range(rounds), disable=not progress, desc="Rounds"):

        # For each edge, decide if it is active this round (bidirectional selection)
        edges = np.array(list(graph.edges()))
        idxs = np.random.choice(len(edges), int(participation_rate * len(edges)), replace=False)
        active_edges = edges[idxs]
        print(active_edges.flatten())
        # Local training (only for clients with a vertex in active_edges)
        active_nodes = set(active_edges.flatten())
        print(f"Fraction of activated nodes : {len(active_nodes)/len(clients)} ")
        train_gradient_norm = 0
        gradients_per_client = {}
        for idx, client in enumerate(clients):
            if idx in active_nodes:
                n_samples, gradient_norm, gradients = client.local_train(local_epochs=local_epochs)
                gradients_per_client[idx] = gradients
                train_gradient_norm += gradient_norm

        # train_results = {
        # 'gradient_norm': train_gradient_norm / max(1, len(active_nodes))
        # }

        # log.info(f"Train, Round {rnd} : gradient_norm : {train_results['gradient_norm']}")
        # metrics['train'].append(train_results)

        ################ OLD GRADIENTS PART ########################
        if old_gradients:
            for active_node in active_nodes:
                neighbors = list(graph.neighbors(active_node))
                neighbor_gradients = {n: gradients_per_client[n] for n in neighbors if n in gradients_per_client}
                neighbor_gradients[int(active_node)] = gradients_per_client[int(active_node)]
                clients[int(active_node)].store_neighbor_gradients(neighbor_gradients)
                

        # # ################## ONLY ACTUALIZE WITH NEW GRADIENTS ########################
        else:
            for active_node in active_nodes:
                neighbors = list(graph.neighbors(active_node))
                neighbor_gradients = {n: gradients_per_client[n] for n in neighbors if n in gradients_per_client}
                neighbor_gradients[int(active_node)] = gradients_per_client[int(active_node)]
                clients[int(active_node)].neighbor_gradients = neighbor_gradients
        ################# MODEL AVERGAE ##################
        # for active_node in active_nodes:
        #     neighbors = list(graph.neighbors(active_node))
        #     neighbors_weights = {n: clients[int(n)].get_state() for n in neighbors if n in gradients_per_client}
        #     neighbors_weights[int(active_node)] = clients[int(active_node)].get_state()
        #     clients[int(active_node)].neighbors_weights = neighbors_weights

        # Apply aggregated shared states
        for node in active_nodes:
            clients[node].update_state(W[node, :], consensus_lr=consensus_lr)

        # Evaluate (average across clients)
        with torch.no_grad():
            total_test_samples = 0
            total_train_samples = 0
            test_weighted_loss = 0.0
            test_weighted_acc = 0.0
            train_weighted_loss = 0.0
            train_weighted_acc = 0.0

            for node in active_nodes:
                client = clients[int(node)]
                ###### TEST METRICS #######
                test_loss, test_acc = client.evaluate()
                test_num_samples = len(client.test_loader.dataset)
                test_weighted_loss += test_loss * test_num_samples
                test_weighted_acc += test_acc * test_num_samples
                total_test_samples += test_num_samples

                ###### TRAIN METRICS #######
                train_loss, train_acc = client.evaluate(client.train_loader)
                train_num_samples = len(client.train_loader.dataset)
                train_weighted_loss += train_loss * train_num_samples
                train_weighted_acc += train_acc * train_num_samples
                total_train_samples += train_num_samples


            test_avg_loss = test_weighted_loss / total_test_samples
            test_avg_acc = test_weighted_acc / total_test_samples
            train_avg_loss = train_weighted_loss / total_train_samples
            train_avg_acc = train_weighted_acc / total_train_samples

            metrics['train']['loss'].append(train_avg_loss)
            metrics['train']['accuracy'].append(train_avg_acc)
            metrics['test']['loss'].append(test_avg_loss)
            metrics['test']['accuracy'].append(test_avg_acc)

            log.info(f"Train, Round {rnd} : loss => {train_avg_loss},  accuracy: {train_avg_acc}")
            log.info(f"Test, Round {rnd} : loss => {test_avg_loss},  accuracy: {test_avg_acc}")

        if rnd >= 5 and max(metrics['test']['accuracy'][-5:]) - min(metrics['test']['accuracy'][-5:]) <= 1e-4:
            return metrics

        if lr_decay != 0:
            for node in active_nodes:
                clients[node].learning_rate *= lr_decay
                
        full_mean, full_std = compute_consensus_distance([client.model.state_dict() for client in clients])
        log.info(f'Mean Distance between models : {full_mean}, Std between models : {full_std}')

    return metrics
