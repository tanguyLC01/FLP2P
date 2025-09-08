import os
from typing import List, Tuple

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from flp2p.client import FLClient
from flp2p.data import build_client_loaders, get_dataset
from flp2p.graph_runner import build_topology, run_rounds, plot_topology
from flp2p.networks.lenet5 import LeNet5
import logging
import pickle

log = logging.getLogger(__name__)

def print_metrics(metrics: List[Tuple[float, float]], mode: str) -> None:
    for r, (loss, acc) in enumerate(metrics):
        log.info(f"{mode}, Round {r+1:03d}: loss={loss:.4f}, acc={acc:.4f}")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu")

    log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Data
    train_ds, test_ds = get_dataset(cfg.data)
    client_loaders = build_client_loaders(
        train_dataset=train_ds,
        test_dataset=test_ds,
        config=cfg,
        save_plot_path=log_path
    )
    
    # Model + Clients
    clients: List[FLClient] = []
    for i in range(cfg.partition.num_clients):
        if cfg.model.name == "lenet5":
            model = LeNet5(cfg.model).to(device)
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
    graph = build_topology(cfg.partition.num_clients, cfg.graph)
    pickle.dump(graph, open(os.path.join(log_path, "graph.pickle"), 'wb'))

    
    plot_topology(graph, 'graph_topology', os.path.join(log_path, "graph_topology.png"))

    # Train
    metrics = run_rounds(
        clients=clients,
        graph=graph,
        rounds=cfg.train.rounds,
        local_epochs=cfg.train.local_epochs,
        progress=cfg.train.progress,
    )

    print_metrics(metrics['train'], 'Train')
    print_metrics(metrics['test'], 'Test')



if __name__ == "__main__":
    main() 