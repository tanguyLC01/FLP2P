import os
from typing import List, Dict

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from flp2p.client import FLClient
from flp2p.data import build_client_loaders, get_dataset
from flp2p.graph_runner import run_rounds
from flp2p.networks.lenet5 import LeNet5
from flp2p.networks.resnet18 import make_resnet18
import logging
import pickle
import random
import numpy as np

from flp2p.utils import plot_topology, build_topology

log = logging.getLogger(__name__)

def print_metrics(metrics: Dict[str, List[float]], mode: str) -> None:
    losses = metrics['loss']
    accuracies = metrics['accuracy']
    rounds = len(metrics[list(metrics.keys())[0]])
    for r, (loss, acc) in enumerate(zip(losses, accuracies)):
        res = f"{mode}, Round {r+1:03d}: "
        res += f'Loss={loss:.4f}, '
        res += f'Accuracy={acc:4f}'
        log.info(res)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu")
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed)   
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        
    log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Data
    train_ds, test_ds = get_dataset(cfg.data)
    client_loaders = build_client_loaders(
        train_dataset=train_ds,
        test_dataset=test_ds,
        config=cfg,
        save_plot_path=log_path
    )
    
    if cfg.model.name == "lenet5":
        base_model = LeNet5(cfg.model).to(device)
        init_state = base_model.state_dict()
    elif cfg.model.name == "resnet18":
        base_model = make_resnet18(cfg.model).to(device)
        init_state = base_model.state_dict()

    # Model + Clients
    clients: List[FLClient] = []
    for i in range(cfg.partition.num_clients):
        if cfg.model.name == "lenet5":
            model = LeNet5(cfg.model).to(device)
            model.load_state_dict(init_state)
        elif cfg.model.name == "resnet18":
            model = make_resnet18(cfg.model).to(device)
            model.load_state_dict(init_state)
        else:
            raise ValueError(f"Unknown model: {cfg.model.name}")
        train_loader, test_loader = client_loaders[i]
        client = FLClient(
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            config=cfg.client
        )
        clients.append(client)
    
    

    # Graph
    graph = build_topology(cfg.partition.num_clients, cfg.graph, mixing_matrix=cfg.mixing_matrix, seed=cfg.seed, consensus_lr=cfg.consensus_lr)
    pickle.dump(graph, open(os.path.join(log_path, "graph.pickle"), 'wb'))
    plot_topology(graph, 'graph_topology', os.path.join(log_path, "graph_topology"))

    # Train
    metrics = run_rounds(
        clients=clients,
        graph=graph,
        mixing_matrix=cfg.mixing_matrix,
        rounds=cfg.train.rounds,
        local_epochs=cfg.train.local_epochs,
        progress=cfg.train.progress,
        participation_rate=cfg.train.participation_rate,
        consensus_lr=cfg.consensus_lr,
        lr_decay=cfg.train.lr_decay,
        old_gradients=cfg.old_gradients
    )

    print_metrics(metrics['train'], 'Train')
    print_metrics(metrics['test'], 'Test')

    return metrics


if __name__ == "__main__":
    main() 