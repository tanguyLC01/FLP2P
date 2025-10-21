
import os
from typing import List, Dict

import torch
from flp2p.client import FLClient
from flp2p.networks.lenet5 import LeNet5
from flp2p.networks.resnet18 import make_resnet18

import numpy as np
import torch
from flp2p.utils import build_topology, compute_weight_matrix
from hydra import compose, initialize

with initialize(version_base=None, config_path="conf", job_name="test_app"):
    cfg = compose(config_name="config")
    
device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu")
base_model = make_resnet18(cfg.model).to(device)

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
W = compute_weight_matrix(graph, cfg.mixing_matrix)

from flp2p.utils import compute_consensus_distance
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

N = len(clients)
for round in range(1, cfg.train.rounds):
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
