import matplotlib.pyplot as plt
from typing import Dict, List, Literal, Tuple

import networkx as nx
import torch
from tqdm import tqdm

from .client import FLClient
import numpy as np
import logging

from flp2p.matcha_mixing_matrix import graphToLaplacian, getProbability, getSubGraphs, getAlpha
from flp2p.data import verify_data_split
from flp2p.utils import compute_weight_matrix, validate_weight_matrix, GOSSIPING
log = logging.getLogger(__name__)


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
    main_link_activation: float = 1,
) -> Dict[str, List[Tuple[float, float]]]:
    metrics: Dict[str, List[Tuple[float, float]]] = {"train": [], 'test': []}

    metrics = {"train":
                                {"loss": [], "accuracy": []},
                    "test":
                                {"loss": [], "accuracy": [], 'std accuracy': []},
            }
    
    log.info("Data split check on random client")
    verify_data_split(np.random.choice(clients, 1)[0])
    
    if mixing_matrix != 'matcha':
        # Compute the mixing matrix
        W = compute_weight_matrix(graph, mixing_matrix)
        validate_weight_matrix(W)
        
        for rnd in tqdm(range(rounds), disable=not progress, desc="Rounds"):

            # For each edge, decide if it is active this round (bidirectional selection)
            edges = list(graph.edges())
            #probs = [data.get('probability_selection', 0) for _, _, data in graph.edges(data=True)]
            #idxs = [idx for idx in range(len(edges)) if np.random.random() < probs[idx]]
            max_degree = sorted(max(graph.degree, key=lambda x: x[1]))
            center_node_1, center_node_2 = max_degree
            active_edges = []
            if np.random.random() < main_link_activation:
                active_edges.append((center_node_1, center_node_2))
                
            edges.remove((center_node_1, center_node_2))
            idxs = np.random.choice(len(edges), int(participation_rate * len(edges)), replace=False)
            active_edges.extend([edges[idx] for idx in idxs])
            log.info(active_edges)

            # Local training
            active_nodes = set(np.array(active_edges).flatten())
            log.info(f"Fraction of activated nodes : {len(active_nodes)/len(clients)} ")
            train_gradient_norm = 0
            weights_per_clients = {}
            for idx, client in enumerate(clients):
                n_samples, gradient_norm, gradients = client.local_train(local_epochs=local_epochs)
                train_gradient_norm += gradient_norm


            train_results = {
            'gradient_norm': train_gradient_norm / max(1, len(active_nodes))
            }

            log.info(f"Train, Round {rnd} : gradient_norm : {train_results['gradient_norm']}")
            # metrics['train'].append(train_results)
            
            ##### RECOMPUTE W #####
            active_subgraph = nx.Graph()
            active_subgraph.add_nodes_from(active_nodes)
            active_subgraph.add_edges_from(active_edges)
            W_active = compute_weight_matrix(active_subgraph, mixing_matrix="metropolis_hasting")

            ################# GOSSIPING PHASE ##################
            for gossip_round in range(1):
                for active_node in active_nodes:
                    neighbors_activated = [v for u, v in edges if u == active_node and v in active_nodes] + [u for u, v in edges if v == active_node and u in active_nodes]
                    neighbor_models = {}
                    for n in neighbors_activated:
                        neighbor_models[n] = clients[n].get_state()
                    neighbor_models[int(active_node)] = clients[active_node].get_state()

                    ################ OLD GRADIENTS PART ########################
                    if old_gradients:
                        clients[int(active_node)].store_neighbor_models(neighbor_models)

                    ################## ONLY ACTUALIZE WITH NEW GRADIENTS ########################
                    else:
                        clients[int(active_node)].neighbor_models = neighbor_models
                        
            ################# AGGREGATION #################
        
                for active_node in active_nodes:
                    clients[active_node].update_state(W_active[active_node, :], consensus_lr=consensus_lr)
                
                    #Evaluate (average across clients)
            with torch.no_grad():
                total_test_samples = 0
                total_train_samples = 0
                test_weighted_loss = 0.0
                test_weighted_acc = 0.0
                train_weighted_loss = 0.0
                train_weighted_acc = 0.0
                accuracies = []
                for client in clients:
                    ###### TEST METRICS #######
                    test_loss, test_acc = client.evaluate()
                    test_num_samples = len(client.test_loader.dataset)
                    test_weighted_loss += test_loss * test_num_samples
                    test_weighted_acc += test_acc * test_num_samples
                    accuracies.append(test_weighted_acc)
                    total_test_samples += test_num_samples

                    ###### TRAIN METRICS #######
                    train_loss, train_acc = client.evaluate(client.train_loader)
                    train_num_samples = len(client.train_loader.dataset)
                    train_weighted_loss += train_loss * train_num_samples
                    train_weighted_acc += train_acc * train_num_samples
                    total_train_samples += train_num_samples


                test_avg_loss = test_weighted_loss / total_test_samples
                test_avg_acc = test_weighted_acc / total_test_samples
                test_std_acc = np.std(np.array(accuracies)/total_test_samples)
                train_avg_loss = train_weighted_loss / total_train_samples
                train_avg_acc = train_weighted_acc / total_train_samples

                metrics['train']['loss'].append(train_avg_loss)
                metrics['train']['accuracy'].append(train_avg_acc)
                metrics['test']['loss'].append(test_avg_loss)
                metrics['test']['accuracy'].append(test_avg_acc)
                metrics['test']['std accuracy'].append(test_std_acc)

                log.info(f"Train, Round {rnd} : loss => {train_avg_loss},  accuracy: {train_avg_acc}")
                log.info(f"Test, Round {rnd} : loss => {test_avg_loss},  accuracy: {test_avg_acc}, std: {test_std_acc}")

            if rnd >= 5 and max(metrics['test']['accuracy'][-10:]) - min(metrics['test']['accuracy'][-10:]) <= 1e-4:
                return metrics

            if lr_decay != 0:
                for client in clients:
                    client.learning_rate *= lr_decay
                consensus_lr *= lr_decay

            max_degree_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)[:2]
            center_node_1, center_node_2 = [n for n, _ in max_degree_nodes]

            # 1️⃣ Collect and flatten all model parameters to CPU
            param_vectors = []
            for client in clients:
                # move each tensor to CPU before flattening
                state = client.model.state_dict()
                flat = torch.cat([p.detach().cpu().flatten() for p in state.values()])
                param_vectors.append(flat)

            param_vectors = torch.stack(param_vectors, dim=0)  # shape [n_clients, d] on CPU
            mean_model = param_vectors.mean(dim=0)

            # 2️⃣ Overall consensus (mean disagreement)
            consensus_distance = torch.mean(torch.norm(param_vectors - mean_model, dim=1)).item()

            # 3️⃣ Cluster-specific consensus (for two stars)
            c1_nodes = list(graph.neighbors(center_node_1)) + [center_node_1]
            c2_nodes = list(graph.neighbors(center_node_2)) + [center_node_2]

            if len(c1_nodes) > 0:
                mean_c1 = param_vectors[c1_nodes].mean(dim=0)
                consensus_distance_c1 = torch.mean(
                    torch.norm(param_vectors[c1_nodes] - mean_c1, dim=1)
                ).item()
            else:
                consensus_distance_c1 = float("nan")

            if len(c2_nodes) > 0:
                mean_c2 = param_vectors[c2_nodes].mean(dim=0)
                consensus_distance_c2 = torch.mean(
                    torch.norm(param_vectors[c2_nodes] - mean_c2, dim=1)
                ).item()
            else:
                consensus_distance_c2 = float("nan")

            # 4️⃣ Distance between cluster means (cross-cluster disagreement)
            if len(c1_nodes) > 0 and len(c2_nodes) > 0:
                center_distance = torch.norm(mean_c1 - mean_c2).item()
            else:
                center_distance = float("nan")

            # 5️⃣ Log metrics
            log.info(f"Overall consensus distance : {consensus_distance:.6f}")
            log.info(f"Cluster 1 consensus distance : {consensus_distance_c1:.6f}")
            log.info(f"Cluster 2 consensus distance : {consensus_distance_c2:.6f}")
            log.info(f"Inter-cluster distance : {center_distance:.6f}")


            # full_mean, full_std = compute_consensus_distance([client.model.state_dict() for client in clients])
            # log.info(f'Mean Distance between models : {full_mean}, Std between models : {full_std}')
                
    elif mixing_matrix == 'matcha':
        W = list()
        edges = np.array(list(graph.edges()))
        n_nodes = len(graph.nodes)
        subgraphs = getSubGraphs(graph, n_nodes)
        laplacians = graphToLaplacian(subgraphs, n_nodes)
        probas = getProbability(laplacians, 2/5)
        alpha = getAlpha(laplacians, probas, n_nodes)
        for _ in range(rounds):
            L_k = np.sum([laplacians[i] for i in range(len(subgraphs)) if np.random.random() < probas[i]], axis=0)
            W.append(np.eye(n_nodes) - alpha * L_k)
            
            
        log.info(f"Mixing Matrix generated, number of matrix :{len(W)} ")
        for rnd, W_actual in enumerate(W):
            
            mask =  ~np.eye(W_actual.shape[0], dtype=bool)
            non_zero_indices  = np.nonzero(W_actual * mask)
            nodes_involved = set(non_zero_indices[0]) | set(non_zero_indices[1])
            log.info(f"Fraction of activated nodes : {len(nodes_involved)/len(clients)} ")
            train_gradient_norm = 0
            weights_per_clients = {}
            for idx, client in enumerate(clients):
                n_samples, gradient_norm, gradients = client.local_train(local_epochs=local_epochs)
                if idx in nodes_involved:
                    weights_per_clients[idx] = client.get_state()
                train_gradient_norm += gradient_norm

            train_results = {
            'gradient_norm': train_gradient_norm / max(1, len(clients))
            }

            log.info(f"Train, Round {rnd} : gradient_norm : {train_results['gradient_norm']}")
            for active_node in nodes_involved:
                neighbors_activated = [v for u, v in edges if W_actual[u, v] != 0]
                neighbor_gradients = {}
                for n in neighbors_activated:
                    neighbor_gradients[n] = weights_per_clients[int(n)]
                neighbor_gradients[int(active_node)] = weights_per_clients[int(active_node)]

                ################ OLD GRADIENTS PART ########################
                if old_gradients:
                    clients[int(active_node)].store_neighbor_gradients(neighbor_gradients)

                ################## ONLY ACTUALIZE WITH NEW GRADIENTS ########################
                else:
                    clients[int(active_node)].neighbor_gradients = neighbor_gradients
                    
            ################# AGGREGATION ##################
            # Apply aggregated shared states
            for active_node in nodes_involved:
                clients[active_node].update_state(W_actual[active_node, :], consensus_lr=consensus_lr)
                
            
            #Evaluate (average across clients)
            with torch.no_grad():
                total_test_samples = 0
                total_train_samples = 0
                test_weighted_loss = 0.0
                test_weighted_acc = 0.0
                train_weighted_loss = 0.0
                train_weighted_acc = 0.0
                accuracies = []
                for client in clients:
                    ###### TEST METRICS #######
                    test_loss, test_acc = client.evaluate()
                    test_num_samples = len(client.test_loader.dataset)
                    test_weighted_loss += test_loss * test_num_samples
                    test_weighted_acc += test_acc * test_num_samples
                    accuracies.append(test_weighted_acc)
                    total_test_samples += test_num_samples

                    ###### TRAIN METRICS #######
                    train_loss, train_acc = client.evaluate(client.train_loader)
                    train_num_samples = len(client.train_loader.dataset)
                    train_weighted_loss += train_loss * train_num_samples
                    train_weighted_acc += train_acc * train_num_samples
                    total_train_samples += train_num_samples


                test_avg_loss = test_weighted_loss / total_test_samples
                test_avg_acc = test_weighted_acc / total_test_samples
                test_std_acc = np.std(np.array(accuracies)/total_test_samples)
                train_avg_loss = train_weighted_loss / total_train_samples
                train_avg_acc = train_weighted_acc / total_train_samples

                metrics['train']['loss'].append(train_avg_loss)
                metrics['train']['accuracy'].append(train_avg_acc)
                metrics['test']['loss'].append(test_avg_loss)
                metrics['test']['accuracy'].append(test_avg_acc)
                metrics['test']['std accuracy'].append(test_std_acc)

                log.info(f"Train, Round {rnd} : loss => {train_avg_loss},  accuracy: {train_avg_acc}")
                log.info(f"Test, Round {rnd} : loss => {test_avg_loss},  accuracy: {test_avg_acc}, std: {test_std_acc}")

            if rnd >= 5 and max(metrics['test']['accuracy'][-10:]) - min(metrics['test']['accuracy'][-10:]) <= 1e-4:
                return metrics

            if lr_decay != 0:
                for client in clients:
                    client.learning_rate *= lr_decay
                #consensus_lr *= lr_decay
                    
                    
            max_degree_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)[:2]
            center_node_1, center_node_2 = [n for n, _ in max_degree_nodes]

            # 1️⃣ Collect and flatten all model parameters to CPU
            param_vectors = []
            for client in clients:
                # move each tensor to CPU before flattening
                state = client.model.state_dict()
                flat = torch.cat([p.detach().cpu().flatten() for p in state.values()])
                param_vectors.append(flat)

            param_vectors = torch.stack(param_vectors, dim=0)  # shape [n_clients, d] on CPU
            mean_model = param_vectors.mean(dim=0)

            # 2️⃣ Overall consensus (mean disagreement)
            consensus_distance = torch.mean(torch.norm(param_vectors - mean_model, dim=1)).item()

            # 3️⃣ Cluster-specific consensus (for two stars)
            c1_nodes = list(graph.neighbors(center_node_1)) + [center_node_1]
            c2_nodes = list(graph.neighbors(center_node_2)) + [center_node_2]

            if len(c1_nodes) > 0:
                mean_c1 = param_vectors[c1_nodes].mean(dim=0)
                consensus_distance_c1 = torch.mean(
                    torch.norm(param_vectors[c1_nodes] - mean_c1, dim=1)
                ).item()
            else:
                consensus_distance_c1 = float("nan")

            if len(c2_nodes) > 0:
                mean_c2 = param_vectors[c2_nodes].mean(dim=0)
                consensus_distance_c2 = torch.mean(
                    torch.norm(param_vectors[c2_nodes] - mean_c2, dim=1)
                ).item()
            else:
                consensus_distance_c2 = float("nan")

            # 4️⃣ Distance between cluster means (cross-cluster disagreement)
            if len(c1_nodes) > 0 and len(c2_nodes) > 0:
                center_distance = torch.norm(mean_c1 - mean_c2).item()
            else:
                center_distance = float("nan")

            # 5️⃣ Log metrics
            log.info(f"Overall consensus distance : {consensus_distance:.6f}")
            log.info(f"Cluster 1 consensus distance : {consensus_distance_c1:.6f}")
            log.info(f"Cluster 2 consensus distance : {consensus_distance_c2:.6f}")
            log.info(f"Inter-cluster distance : {center_distance:.6f}")


            # full_mean, full_std = compute_consensus_distance([client.model.state_dict() for client in clients])
            # log.info(f'Mean Distance between models : {full_mean}, Std between models : {full_std}')

    return metrics
