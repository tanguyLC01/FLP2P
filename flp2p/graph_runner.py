import matplotlib.pyplot as plt
from typing import Dict, List, Literal, Tuple

import networkx as nx
from sklearn import metrics
import torch
from tqdm import tqdm

from .client import FLClient
import numpy as np
import logging


from flp2p.matcha_mixing_matrix import graphToLaplacian, getProbability, getSubGraphs, getAlpha
from flp2p.data import verify_data_split
from flp2p.utils import compute_weight_matrix, validate_weight_matrix, GOSSIPING, get_spectral_gap
log = logging.getLogger(__name__)


class graph_runner:

    def __init__(self,   clients: List[FLClient],
        graph: nx.Graph,
        mixing_matrix: GOSSIPING,
        rounds: int = 5,
        local_epochs: int = 1,
        progress: bool = True,
        old_gradients: bool = True,
        client_config: Dict = {},
        topology_type: str = "two_clusters",
        aggregation_step_per_round: int = 1) -> None:
        self.clients = clients
        self.graph = graph
        self.mixing_matrix = mixing_matrix
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.progress = progress
        self.old_gradients = old_gradients
        self.client_config = client_config
        self.topology_type = topology_type
        self.aggregation_step_per_round = aggregation_step_per_round
        
        self.metrics: Dict[str, List[Tuple[float, float]]] = {"train": [], 'test': []}
        self.metrics = {"train":
                                    {"loss": [], "accuracy": []},
                        "test":    
                                    {"loss": [], "accuracy": [], 'std accuracy': []},
        }
        
        self.weights_per_clients: Dict[int, Dict[str, torch.Tensor]] = {}
        
        if self.topology_type == "two_clusters":
            max_degree_nodes = list(sorted(graph.degree, key=lambda x: x[1], reverse=True)[:2])
            self.center_node_1, self.center_node_2 = [n for n, _ in max_degree_nodes]  
            self.neighbor_center_1 = list([n for n in graph.neighbors(self.center_node_1) if n != self.center_node_2])[0]
            self.neighbor_center_2 = list([n for n in graph.neighbors(self.center_node_2) if n != self.center_node_1])[0]
            main_link_proba = graph[self.center_node_1][self.center_node_2]['probability_selection']
            border_link_proba = graph[self.center_node_1][self.neighbor_center_1]['probability_selection']
            log.info(f'Border_link_activation : {border_link_proba}')
            log.info(f'Main_link_activation : {main_link_proba}')

        

    def lr_update(self, rnd: int) -> None:
        if "lr_schedule" in self.client_config:
            client = self.clients[0]
            if (rnd+1) <= self.client_config.lr_schedule.warmup.epochs:
                for client in self.clients:
                    lr = (self.client_config.learning_rate - self.client_config.lr_schedule.warmup.start_lr) *  (rnd+1) / float(self.client_config.lr_schedule.warmup.epochs) + self.client_config.lr_schedule.warmup.start_lr
                    for client in self.clients:
                        client.learning_rate = lr 
            else:
                if (rnd+1) in self.client_config.lr_schedule.decay_milestones:
                    lr = self.client_config.lr_schedule.factor
                    for client in self.clients:
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
        
    
    
    def training(self, rnd: int) -> None:
        # Local training of all the nodes
        train_gradient_norm = 0
        
        for idx, client in enumerate(self.clients):
            _, gradient_norm, _ = client.local_train(local_epochs=self.local_epochs)
            self.weights_per_clients[idx] = client.get_state()
            train_gradient_norm += gradient_norm

        train_results = {
        'gradient_norm': train_gradient_norm / max(1, len(self.clients))
        }

        log.info(f"Train, Round {rnd} : gradient_norm : {train_results['gradient_norm']}")
        
    
    def gossip_phase(self, W: np.array) -> None:
        mask =  ~np.eye(W.shape[0], dtype=bool)
        communicative_nodes = np.nonzero(W * mask)
        nodes_involved = set(communicative_nodes[0])
        log.info(f"Nodes involved : {nodes_involved}")
        for idx_client in range(len(self.clients)):
            client = self.clients[idx_client]
            if client in nodes_involved:
                neighbors_activated =  np.where(W[idx_client, :] > 0)[0]
                neighbor_models = {n: self.weights_per_clients[int(n)] for n in neighbors_activated}

                ################ OLD GRADIENTS PART ########################
                if self.old_gradients:
                    client.store_neighbor_gradients(neighbor_models)

                ################## ONLY ACTUALIZE WITH NEW GRADIENTS ########################
                else:
                    client.neighbor_models = neighbor_models
                
                W = np.linalg.matrix_power(W, self.aggregation_step_per_round)
                # We can do the update right now because neihborgs take model that are frozen on the CPU. Hence, the new update will not be seen by the next model and we do not insert any asynchronous update.            
                self.clients[idx_client].update_state(W[idx_client, :])
                
            else:
                client.neighbor_models = {}
                
    def load_metrics(self, rnd: int) -> None:
        #Evaluate (average across clients)
        with torch.no_grad():
            total_test_samples = 0
            total_train_samples = 0
            test_weighted_loss = 0.0
            test_weighted_acc = 0.0
            train_weighted_loss = 0.0
            train_weighted_acc = 0.0
            accuracies = []
            for client in self.clients:
                ###### TEST METRICS #######
                test_loss, test_acc = client.evaluate()
                test_num_samples = len(client.test_loader.dataset)
                test_weighted_loss += test_loss * test_num_samples
                test_weighted_acc += test_acc * test_num_samples
                accuracies.append(test_acc)
                total_test_samples += test_num_samples

                ###### TRAIN METRICS #######
                train_loss, train_acc = client.evaluate(client.train_loader)
                train_num_samples = len(client.train_loader.dataset)
                train_weighted_loss += train_loss * train_num_samples
                train_weighted_acc += train_acc * train_num_samples
                total_train_samples += train_num_samples

            test_avg_loss = test_weighted_loss / total_test_samples
            test_avg_acc = test_weighted_acc / total_test_samples
            test_std_acc = np.std(accuracies)
            train_avg_loss = train_weighted_loss / total_train_samples
            train_avg_acc = train_weighted_acc / total_train_samples

            self.metrics['train']['loss'].append(train_avg_loss)
            self.metrics['train']['accuracy'].append(train_avg_acc)
            self.metrics['test']['loss'].append(test_avg_loss)
            self.metrics['test']['accuracy'].append(test_avg_acc)
            self.metrics['test']['std accuracy'].append(test_std_acc)

            log.info(f"Train, Round {rnd} : loss => {train_avg_loss},  accuracy: {train_avg_acc}")
            log.info(f"Test, Round {rnd} : loss => {test_avg_loss},  accuracy: {test_avg_acc}, std: {test_std_acc}")
            
        param_vectors = []
        for client in self.clients:
            # move each tensor to CPU before flattening
            state = client.model.state_dict()
            flat = torch.cat([p.detach().cpu().flatten() for p in state.values()])
            param_vectors.append(flat)

        param_vectors = torch.stack(param_vectors, dim=0)  # shape [n_clients, d] on CPU
        mean_model = param_vectors.mean(dim=0)
        
        # Overall consensus (mean disagreement)
        consensus_distance = torch.mean(torch.norm(param_vectors - mean_model, dim=1)).item()

        if self.topology_type == "two_clusters":
            # Inter-cluster consensus distances
            inter_cluster = torch.norm(param_vectors[self.center_node_1] - param_vectors[self.center_node_2]).item()

            # Intra-cluster consensus distances
            cluster_1_consensus_distance = torch.norm(param_vectors[self.center_node_1] - param_vectors[self.neighbor_center_1]).item()
            cluster_2_consensus_distance = torch.norm(param_vectors[self.center_node_2] - param_vectors[self.neighbor_center_2]).item()
            log.info(f'Number of model stored in central node 1 : {len(self.clients[self.center_node_1].neighbor_models)}')
            
            
        log.info(f"Overall consensus distance : {consensus_distance:.6f}")
        if self.topology_type == "two_clusters":
            log.info(f"Cluster 1 consensus distance : {cluster_1_consensus_distance:.6f}")
            log.info(f"Cluster 2 consensus distance : {cluster_2_consensus_distance:.6f}")
            log.info(f"Inter-cluster distance : {inter_cluster:.6f}")
       
       
                
    def run(
        self
    ) -> Dict[str, List[Tuple[float, float]]]:
        
        log.info("Data split check on random client")
        verify_data_split(np.random.choice(self.clients, 1)[0])
            
        if self.old_gradients:
            # If old_gradiens is True, fill the neighbord_models with the x_0 of the FL Client just created
            for i, client in enumerate(self.clients):
                client.neighbor_models = {n: self.clients[n].get_state() for n in self.graph.neighbors(i)}

        if self.mixing_matrix != 'matcha':
            # Compute the mixing matrix
            W = compute_weight_matrix(self.graph, self.mixing_matrix)
            validate_weight_matrix(W)
            
            N = len(self.clients)
            for rnd in tqdm(range(self.rounds), disable=not self.progress, desc="Rounds"):
                log.info(f'-------------- Round {rnd} --------------')
                self.lr_update(rnd)

                # For each edge, decide if it is active this round (bidirectional selection)
                g_temp = self.  graph.copy()
                if self.topology_type == "two_clusters":
                    border_nodes = [n for n in self.graph.nodes if self.graph.degree[n] == 1]
                    if np.random.random() > self.graph[self.center_node_1][self.center_node_2]["probability_selection"]:
                        g_temp.remove_edge(self.center_node_1, self.center_node_2)
                    for border_node in border_nodes:
                        if g_temp.has_edge(border_node, self.center_node_1) and  np.random.random() > self.graph[self.center_node_1][border_node]["probability_selection"]:
                            g_temp.remove_edge(border_node, self.center_node_1)
                        elif g_temp.has_edge(border_node, self.center_node_2) and np.random.random() > self.graph[border_node][self.center_node_2]["probability_selection"]:
                            g_temp.remove_edge(border_node, self.center_node_2)

                # Show the number of active nodes in the generated subgraph
                active_nodes = [n for n in g_temp.nodes if g_temp.degree[n] >= 1]
                log.info(f"Fraction of activated nodes : {len(active_nodes)/len(self.clients)} ") 
                        
                ################# GOSSIPING PHASE ################
                self.training(rnd=rnd)
                            
                if not self.old_gradients:
                    ##### RECOMPUTE W #####
                    W_active = compute_weight_matrix(g_temp, mixing_matrix=self.mixing_matrix)
                else:
                    W_active = W
                    
                validate_weight_matrix(W_active)
            
                ################# GOSSIPING PHASE ##################
                self.gossip_phase(W_active)
                
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

                
                # Evaluate (average across clients)
                self.load_metrics(rnd)

                # full_mean, full_std = compute_consensus_distance([client.model.state_dict() for client in clients])
                # log.info(f'Mean Distance between models : {full_mean}, Std between models : {full_std}')
                    
        elif self.mixing_matrix == 'matcha':
            W = list()
            n_nodes = len(self.graph.nodes)
            subgraphs = getSubGraphs(self.graph, n_nodes)
            laplacians = graphToLaplacian(subgraphs, n_nodes)
            probas = getProbability(laplacians, 2/5)
            alpha = getAlpha(laplacians, probas, n_nodes)
            log.info(f"alpha : {alpha}")
            for _ in range(self.rounds):
                L_k = np.sum([laplacians[i] for i in range(len(subgraphs)) if np.random.random() < probas[i]], axis=0)
                W.append(np.eye(n_nodes) - alpha * L_k)
                
                
            log.info(f"Mixing Matrix generated, number of matrix :{len(W)} ")
            for rnd, W_actual in enumerate(W):
                
                log.info(f'-------------- Round {rnd} --------------')
                log.info(f"Spectral Gap : {get_spectral_gap(W_actual)}")
                self.lr_update(rnd)
                
                ############ TRAINING PHASE ##############
                # Local training of all the nodes
                self.training(rnd=rnd)
                
                ############## GOSSIPING PHASE ##############
                self.gossip_phase(W_actual)
                

                # Evaluate (average across clients)
                self.load_metrics(rnd)

                # full_mean, full_std = compute_consensus_distance([client.model.state_dict() for client in clients])
                # log.info(f'Mean Distance between models : {full_mean}, Std between models : {full_std}')

        return self.load_metrics
