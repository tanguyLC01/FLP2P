
import os
from typing import List

import torch
from flp2p.client import FLClient
from flp2p.networks.lenet5 import LeNet5
from flp2p.networks.resnet18 import make_resnet18
from omegaconf import DictConfig
import numpy as np
import torch
from flp2p.utils import build_topology, compute_weight_matrix, plot_topology, validate_weight_matrix
import pickle
import logging
import hydra
import random
from flp2p.matcha_mixing_matrix import getAlpha, getProbability, getSubGraphs, graphToLaplacian
log = logging.getLogger(__name__)


def get_flat_params(model):
    """Flatten model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    """Load a flat parameter vector into a model."""
    idx = 0
    for p in model.parameters():
        num_params = p.numel()
        p.data.copy_(flat_params[idx:idx + num_params].view_as(p))
        idx += num_params
        
def get_spectral_gap(matrix: np.array) -> float:
    eigenvals = set(np.abs(np.linalg.eigvals(matrix)))
    if len(eigenvals) == 1:
        return 1
    lambda_2 = sorted(eigenvals, reverse=False)[1]
    return 1 - lambda_2

@hydra.main(version_base=None, config_path="conf", config_name="fixed")
def run_fixed(cfg: DictConfig) -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu")
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed)   
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


    # Model + Clients
    clients: List[FLClient] = []
    for i in range(cfg.partition.num_clients):
        if cfg.model.name == "lenet5":
            model = LeNet5(cfg.model).to(device)
        elif cfg.model.name == "resnet18":
            model = make_resnet18(cfg.model).to(device)
        else:
            raise ValueError(f"Unknown model: {cfg.model.name}")
        train_loader, test_loader = None, None
        client = FLClient(
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            config=cfg.client
        )
        clients.append(client)
        
        
    graph = build_topology(cfg.partition.num_clients, cfg.graph, mixing_matrix=cfg.mixing_matrix, seed=cfg.seed, consensus_lr=cfg.consensus_lr)
    pickle.dump(graph, open(os.path.join(log_path, "graph.pickle"), 'wb'))
    plot_topology(graph, 'graph_topology', os.path.join(log_path, "graph_topology"))
    
        
    if cfg.old_gradients:
        # If old_gradiens is True, fill the neighbord_models with the x_0 of the FL Client just created
        for i, client in enumerate(clients):
            client.neighbor_models = {n: clients[n].get_state() for n in graph.neighbors(i)}
    
    
    N = len(clients)
    max_degree_nodes = list(sorted(graph.degree, key=lambda x: x[1], reverse=True)[:2])
    center_node_1, center_node_2 = [n for n, _ in max_degree_nodes]  
    neighbor_center_1 = list([n for n in graph.neighbors(center_node_1) if n != center_node_2])[0]
    neighbor_center_2 = list([n for n in graph.neighbors(center_node_2) if n != center_node_1])[0] 
    log.info(f'Center_node_1 : {center_node_1}, Center_node_2 : {center_node_2}, Neighbord_1 : {neighbor_center_1}, Neighbor_2 : {neighbor_center_2}')

    if cfg.mixing_matrix != 'matcha':
        W = compute_weight_matrix(graph, cfg.mixing_matrix)
    elif cfg.mixing_matrix == 'matcha':
        W = compute_weight_matrix(graph, 'jaccard') # We use this W for the aggregation part and with old_models updates, we need a full neighbor matrix
    if cfg.mixing_matrix != 'matcha':
        W_list = list()
        border_nodes = [n for n in graph.nodes if graph.degree[n] == 1]
        log.info(f'Number of rounds : {cfg.train.rounds}')
        for _ in range(cfg.train.rounds):
            # Copy base graph
            g_temp = graph.copy()

            # Randomly desactivate the main link
            if np.random.random() > cfg.main_link_activation:
                if g_temp.has_edge(center_node_1, center_node_2):
                    g_temp.remove_edge(center_node_1, center_node_2)

            # Randomly desactivate border links
            for border_node in border_nodes:
                if np.random.random() > cfg.border_link_activation:
                    if g_temp.has_edge(border_node, center_node_1):
                        g_temp.remove_edge(border_node, center_node_1)
                    if g_temp.has_edge(border_node, center_node_2):
                        g_temp.remove_edge(border_node, center_node_2)
                        
            temp = compute_weight_matrix(g_temp, cfg.mixing_matrix)
            log.info(f"Rounds {_+1}, Spectral Gap = {get_spectral_gap(temp)}")
            validate_weight_matrix(temp)
            W_list.append(temp)
        
    else:
        W_list = list()
        n_nodes = len(graph.nodes)
        subgraphs = getSubGraphs(graph, n_nodes)
        laplacians = graphToLaplacian(subgraphs, n_nodes)
        probas = getProbability(laplacians, 2/5)
        alpha = getAlpha(laplacians, probas, n_nodes)
        for _ in range(cfg.train.rounds):
            L_k = np.sum([laplacians[i] for i in range(len(subgraphs)) if np.random.random() < probas[i]], axis=0)
            temp = np.eye(n_nodes) - alpha * L_k
            log.info(f"Rounds {_+1}, Spectral Gap = {get_spectral_gap(temp)}")
            W_list.append(temp)


    for round in range(1, cfg.train.rounds):
        W_actual = W_list[round-1]
            
        with torch.no_grad():
            mask = ~np.eye(W_actual.shape[0], dtype=bool)
            non_zero_indices  = np.nonzero(W_actual * mask)
            nodes_involved = set(non_zero_indices[0]) | set(non_zero_indices[1])
            for active_node in nodes_involved:
                neighbors_activated = [int(n) for n in graph.neighbors(active_node) if W_actual[active_node, n] > 0]
                if active_node not in neighbors_activated:
                    neighbors_activated.append(int(active_node))
                neighbor_models = {}
                for n in neighbors_activated:
                    neighbor_models[n] = clients[n].get_state()

                ################ OLD GRADIENTS PART ########################
                if cfg.old_gradients:
                    clients[int(active_node)].store_neighbor_models(neighbor_models)

                ################## ONLY ACTUALIZE WITH NEW GRADIENTS ########################
                else:
                    clients[int(active_node)].neighbor_models = neighbor_models 
            
            W_mixing = None
            if not cfg.old_gradients:
                W_mixing = W_actual
            else:
                W_mixing = W
            for active_node in nodes_involved:
                clients[active_node].update_state(W_mixing[active_node, :])               
        
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
            inter_cluster = torch.norm(param_vectors[center_node_1] - param_vectors[center_node_2]).item()

            cluster_1_consensus_distance = torch.norm(param_vectors[center_node_1] - param_vectors[neighbor_center_1]).item()
            cluster_2_consensus_distance = torch.norm(param_vectors[center_node_2] - param_vectors[neighbor_center_2]).item()
            
            
            mass = torch.mean(torch.norm(param_vectors, dim=1)).item()
            log.info(f'-------------- Round {round} --------------')
            log.info(f"Overall consensus distance : {consensus_distance:.6f}")
            log.info(f"Cluster 1 consensus distance : {cluster_1_consensus_distance:.6f}")
            log.info(f"Cluster 2 consensus distance : {cluster_2_consensus_distance:.6f}")
            log.info(f"Inter-cluster distance : {inter_cluster:.6f}")
            log.info(f"Mass : {mass}")


    
if __name__ == "__main__":
    run_fixed()