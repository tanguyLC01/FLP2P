
import os
from typing import List

import torch
from flp2p.client import FLClient
from flp2p.networks.lenet5 import LeNet5
from flp2p.networks.resnet18 import make_resnet18
from omegaconf import DictConfig
import numpy as np
import torch
from flp2p.utils import build_topology, compute_weight_matrix, plot_topology
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

@hydra.main(version_base=None, config_path="conf", config_name="config")
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
    if cfg.mixing_matrix != 'matcha':
        W = compute_weight_matrix(graph, cfg.mixing_matrix)
    else:
        W = list()
        n_nodes = len(graph.nodes)
        subgraphs = getSubGraphs(graph, n_nodes)
        laplacians = graphToLaplacian(subgraphs, n_nodes)
        probas = getProbability(laplacians, 2/5)
        alpha = getAlpha(laplacians, probas, n_nodes)
        for _ in range(cfg.train.rounds):
            L_k = np.sum([laplacians[i] for i in range(len(subgraphs)) if np.random.random() < probas[i]], axis=0)
            W.append(np.eye(n_nodes) - alpha * L_k)

    N = len(clients)
    max_degree_nodes = list(sorted(graph.degree, key=lambda x: x[1], reverse=True)[:2])
    center_node_1, center_node_2 = [n for n, _ in max_degree_nodes]  
    neighbor_center_1 = list(graph.neighbors(center_node_1))[0]
    neighbor_center_2 = list(graph.neighbors(center_node_2))[0] 

    for round in range(1, cfg.train.rounds):
        W_actual = None
        if type(W) == list:
            W_actual = W[round-1]
        else:
            W_actual = W
            
        with torch.no_grad():
            flat_params  = [get_flat_params(client.model) for client in clients]
            new_params = []
            
            for i in range(N):
                mixed = torch.zeros_like(flat_params[0])
                for j in range(N):
                    mixed += W[i, j] * flat_params[j]
                new_params.append(mixed)
            
            # Update each client model
            for i in range(N):
                set_flat_params(clients[i].model, new_params[i])
                
        
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
            
            log.info(f'-------------- Round {round} --------------')
            log.info(f"Overall consensus distance : {consensus_distance:.6f}")
            log.info(f"Cluster 1 consensus distance : {cluster_1_consensus_distance:.6f}")
            log.info(f"Cluster 2 consensus distance : {cluster_2_consensus_distance:.6f}")
            log.info(f"Inter-cluster distance : {inter_cluster:.6f}")


    
if __name__ == "__main__":
    run_fixed()