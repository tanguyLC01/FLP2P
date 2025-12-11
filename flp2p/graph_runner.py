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
from flp2p.utils import compute_weight_matrix, validate_weight_matrix, GOSSIPING, get_spectral_gap
log = logging.getLogger(__name__)

def lr_update(client_config: dict, rnd: int, clients: List[FLClient]):
        if "lr_schedule" in client_config:
            client = clients[0]
            if (rnd+1) <= client_config.lr_schedule.warmup.epochs:
                for client in clients:
                    lr = (client_config.learning_rate - client_config.lr_schedule.warmup.start_lr) *  (rnd+1) / float(client_config.lr_schedule.warmup.epochs) + client_config.lr_schedule.warmup.start_lr
                    for client in clients:
                        client.learning_rate = lr 
            else:
                if (rnd+1) in client_config.lr_schedule.decay_milestones:
                    lr = client_config.lr_schedule.factor
                    for client in clients:
                        client.learning_rate *= lr
            log.info(f'Lr : {client.learning_rate}')
        # if lr_decay != 0:
        #     if type(lr_decay) is int and rnd > lr_decay:
        #         log.info(f'Learning rate: {learning_rate / (rnd - lr_decay + 1)**1.5}')
        #         for client in clients:
        #             client.learning_rate = learning_rate / (rnd - lr_decay + 1)**1.5
                
        #     if lr_decay is float:
        #         for client in clients:
        #             client.learning_rate *= lr_decay


def run_rounds(
    clients: List[FLClient],
    graph: nx.Graph,
    mixing_matrix: GOSSIPING,
    rounds: int = 5,
    local_epochs: int = 1,
    progress: bool = True,
    lr_decay: int = 0,
    old_gradients: bool = True,
    client_config: Dict = {},
    topology_type: str = "twoc_clusters"
) -> Dict[str, List[Tuple[float, float]]]:
    metrics: Dict[str, List[Tuple[float, float]]] = {"train": [], 'test': []}

    metrics = {"train":
                                {"loss": [], "accuracy": []},
                    "test":
                                {"loss": [], "accuracy": [], 'std accuracy': []},
            }
    
    log.info("Data split check on random client")
    verify_data_split(np.random.choice(clients, 1)[0])
    learning_rate = clients[0].learning_rate
    
    if old_gradients:
        # If old_gradiens is True, fill the neighbord_models with the x_0 of the FL Client just created
        for i, client in enumerate(clients):
            client.neighbor_models = {n: clients[n].get_state() for n in graph.neighbors(i)}
    
    if topology_type == "two_clusters":
        max_degree_nodes = list(sorted(graph.degree, key=lambda x: x[1], reverse=True)[:2])
        center_node_1, center_node_2 = [n for n, _ in max_degree_nodes]  
        neighbor_center_1 = list([n for n in graph.neighbors(center_node_1) if n != center_node_2])[0]
        neighbor_center_2 = list([n for n in graph.neighbors(center_node_2) if n != center_node_1])[0]
        main_link_proba = graph[center_node_1][center_node_2]['probability_selection']
        border_link_proba = graph[center_node_1][neighbor_center_1]['probability_selection']
        log.info(f'Border_link_activation : {border_link_proba}')
        log.info(f'Main_link_activation : {main_link_proba}')

    if mixing_matrix != 'matcha':
        # Compute the mixing matrix
        W = compute_weight_matrix(graph, mixing_matrix)
        validate_weight_matrix(W)
        
        N = len(clients)
        for rnd in tqdm(range(rounds), disable=not progress, desc="Rounds"):
            log.info(f'-------------- Round {rnd} --------------')
            lr_update(client_config, rnd, clients)

            # For each edge, decide if it is active this round (bidirectional selection)
            g_temp = graph.copy()
            if topology_type == "two_clusters":
                border_nodes = [n for n in graph.nodes if graph.degree[n] == 1]
                if np.random.random() > graph[center_node_1][center_node_2]["probability_selection"]:
                    g_temp.remove_edge(center_node_1, center_node_2)

                for border_node in border_nodes:
                    if g_temp.has_edge(border_node, center_node_1) and  np.random.random() > graph[center_node_1][border_node]["probability_selection"]:
                        g_temp.remove_edge(border_node, center_node_1)
                    elif g_temp.has_edge(border_node, center_node_2) and np.random.random() > graph[border_node][center_node_2]["probability_selection"]:
                        g_temp.remove_edge(border_node, center_node_2)

            # Show the number of active nodes in the generated subgraph
            active_nodes = [n for n in g_temp.nodes if g_temp.degree[n] >= 1]
            log.info(f"Fraction of activated nodes : {len(active_nodes)/len(clients)} ") 

            # Local training of all the nodes
            train_gradient_norm = 0
            weights_per_clients = {}
            for idx, client in enumerate(clients):
                _, gradient_norm, _ = client.local_train(local_epochs=local_epochs)
                train_gradient_norm += gradient_norm


            train_results = {
            'gradient_norm': train_gradient_norm / max(1, len(clients))
            }

            log.info(f"Train, Round {rnd} : gradient_norm : {train_results['gradient_norm']}")
            # metrics['train'].append(train_results)
            
                        
            if not old_gradients:
                ##### RECOMPUTE W #####
                W_active = compute_weight_matrix(g_temp, mixing_matrix=mixing_matrix)
            else:
                W_active = W
                
            validate_weight_matrix(W_active)
            ################# GOSSIPING PHASE ##################
            for active_node in active_nodes:
                neighbors_activated = [int(n) for n in g_temp.neighbors(active_node)]
                # We add the node itself in the neighbors set to take it into account the aggregation process 
                if active_node not in neighbors_activated:
                    neighbors_activated.append(int(active_node))
                neighbor_models = {}
                for n in neighbors_activated:
                    neighbor_models[n] = clients[n].get_state()

                ################ OLD GRADIENTS PART ########################
                if old_gradients:
                    clients[int(active_node)].store_neighbor_models(neighbor_models)

                ################## ONLY ACTUALIZE WITH NEW GRADIENTS ########################
                else:
                    clients[int(active_node)].neighbor_models = neighbor_models
                    
            ######## WEIGHT W BY THE ACTIVATION PROBABILITY OF EDEGES #############
            #weights = np.full_like(W_active, 1.0 / border_link_proba)

            # Fix the diagonal to 1 (no scaling)
            # np.fill_diagonal(weights, 1.0)
            # weights[center_node_1, center_node_2] *= 1/main_link_proba
            # weights[center_node_2, center_node_1] *= 1/main_link_proba
            
            # W_active *= weights
            # row_sums = W_active.sum(axis=1, keepdims=True)
            # # Avoid division by zero
            # row_sums[row_sums == 0] = 1.0
            # W_active /= row_sums
            
            ################# AGGREGATION #################
            #log.info(f'Number of model stored in central node 1 : {len(clients[center_node_1].neighbor_models)}')
            for active_node in active_nodes:
                clients[active_node].update_state(W_active[active_node, :])
            
            # Evaluate (average across clients)
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

        
            param_vectors = []
            for client in clients:
                # move each tensor to CPU before flattening
                state = client.model.state_dict()
                flat = torch.cat([p.detach().cpu().flatten() for p in state.values()])
                param_vectors.append(flat)

            param_vectors = torch.stack(param_vectors, dim=0)  # shape [n_clients, d] on CPU
            mean_model = param_vectors.mean(dim=0)
            
            # Overall consensus (mean disagreement)
            consensus_distance = torch.mean(torch.norm(param_vectors - mean_model, dim=1)).item()

            # Inter-cluster consensus distances
            #inter_cluster = torch.norm(param_vectors[center_node_1] - param_vectors[center_node_2]).item()

            #cluster_1_consensus_distance = torch.norm(param_vectors[center_node_1] - param_vectors[neighbor_center_1]).item()
            #cluster_2_consensus_distance = torch.norm(param_vectors[center_node_2] - param_vectors[neighbor_center_2]).item()
            
            log.info(f"Overall consensus distance : {consensus_distance:.6f}")
            # log.info(f"Cluster 1 consensus distance : {cluster_1_consensus_distance:.6f}")
            # log.info(f"Cluster 2 consensus distance : {cluster_2_consensus_distance:.6f}")
            # log.info(f"Inter-cluster distance : {inter_cluster:.6f}")


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
        log.info(f"alpha : {alpha}")
        for _ in range(rounds):
            L_k = np.sum([laplacians[i] for i in range(len(subgraphs)) if np.random.random() < probas[i]], axis=0)
            W.append(np.eye(n_nodes) - alpha * L_k)
            
            
        log.info(f"Mixing Matrix generated, number of matrix :{len(W)} ")
        for rnd, W_actual in enumerate(W):
            
            log.info(f'-------------- Round {rnd} --------------')
            log.info(f"Spectral Gap : {get_spectral_gap(W_actual)}")
            lr_update(client_config, rnd, clients)
            
            mask =  ~np.eye(W_actual.shape[0], dtype=bool)
            non_zero_indices  = np.nonzero(W_actual * mask)
            nodes_involved = set(non_zero_indices[0]) | set(non_zero_indices[1])
            log.info(f"Fraction of activated nodes : {len(nodes_involved)/len(clients)} ")
            train_gradient_norm = 0
            weights_per_clients = {}
            for idx, client in enumerate(clients):
                _, gradient_norm, _ = client.local_train(local_epochs=local_epochs)
                weights_per_clients[idx] = client.get_state()
                train_gradient_norm += gradient_norm

            train_results = {
            'gradient_norm': train_gradient_norm / max(1, len(clients))
            }

            log.info(f"Train, Round {rnd} : gradient_norm : {train_results['gradient_norm']}")
            for active_node in nodes_involved:
                neighbors_activated = [v for u, v in edges if W_actual[u, v] > 0]
                neighbor_models = {}
                for n in neighbors_activated:
                    neighbor_models[n] = weights_per_clients[int(n)]
                neighbor_models[int(active_node)] = weights_per_clients[int(active_node)]

                ################ OLD GRADIENTS PART ########################
                if old_gradients:
                    clients[int(active_node)].store_neighbor_gradients(neighbor_models)

                ################## ONLY ACTUALIZE WITH NEW GRADIENTS ########################
                else:
                    clients[int(active_node)].neighbor_models = neighbor_models
            
            if topology_type == "two_clusters":
                log.info(f'Number of model stored in central node 1 : {len(clients[center_node_1].neighbor_models)}')
            ################# AGGREGATION ##################
            # Apply aggregated shared states
            for active_node in nodes_involved:
                clients[active_node].update_state(W_actual[active_node, :])
            
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
                
            param_vectors = []
            for client in clients:
                # move each tensor to CPU before flattening
                state = client.model.state_dict()
                flat = torch.cat([p.detach().cpu().flatten() for p in state.values()])
                param_vectors.append(flat)

            param_vectors = torch.stack(param_vectors, dim=0)  # shape [n_clients, d] on CPU
            mean_model = param_vectors.mean(dim=0)
            
            # Overall consensus (mean disagreement)
            consensus_distance = torch.mean(torch.norm(param_vectors - mean_model, dim=1)).item()

            if topology_type == "two_clusters":
            # Inter-cluster consensus distances
                inter_cluster = torch.norm(param_vectors[center_node_1] - param_vectors[center_node_2]).item()

                cluster_1_consensus_distance = torch.norm(param_vectors[center_node_1] - param_vectors[neighbor_center_1]).item()
                cluster_2_consensus_distance = torch.norm(param_vectors[center_node_2] - param_vectors[neighbor_center_2]).item()
                
            log.info(f"Overall consensus distance : {consensus_distance:.6f}")
            if topology_type == "two_clusters":
                log.info(f"Cluster 1 consensus distance : {cluster_1_consensus_distance:.6f}")
                log.info(f"Cluster 2 consensus distance : {cluster_2_consensus_distance:.6f}")
                log.info(f"Inter-cluster distance : {inter_cluster:.6f}")


            # full_mean, full_std = compute_consensus_distance([client.model.state_dict() for client in clients])
            # log.info(f'Mean Distance between models : {full_mean}, Std between models : {full_std}')

    return metrics
