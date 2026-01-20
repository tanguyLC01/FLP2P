import numpy as np
import random
import flp2p  # Assuming the flp2p module is installed and configured
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

from flp2p.data import build_client_loaders, get_dataset
from typing import List
from flp2p.networks.lenet5 import LeNet5
from flp2p.networks.resnet18 import make_resnet18
from flp2p.client import FLClient
import logging
from flp2p.utils import lr_update
from main import print_metrics
import random

"""
This file implements a simplified classical FedAvg simulation
using the flp2p framework style. In this example, each client
performs a dummy local update, and the server aggregates the
local models by averaging their weights.
"""

log = logging.getLogger(__name__)

# Define a Server class that handles aggregation
class Server:
    def __init__(self, model):
        self.model = model

    def aggregate(self, client_state_dicts, weights):
        """
        Aggregate client models by weighted averaging of their parameters.
        """
        # Initialize an empty state dict with zeros
        avg_state_dict = {}
        for key in client_state_dicts[0]:
            avg_state_dict[key] = torch.zeros_like(client_state_dicts[0][key])

        # Weighted sum of client parameters
        for state_dict, weight in zip(client_state_dicts, weights):
            for key in avg_state_dict:
                avg_state_dict[key] += state_dict[key] * weight

        self.model.load_state_dict(avg_state_dict)
        return avg_state_dict

@hydra.main(version_base=None, config_path="conf", config_name="config")
def simulate_fedavg(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu")
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed)   
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
          
    log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir # type: ignore
    train_ds, test_ds = get_dataset(cfg.data)
    client_loaders = build_client_loaders(
        train_dataset=train_ds,
        test_dataset=test_ds,
        config=cfg,
        save_plot_path=log_path
    )
    
    init_state = {}
    base_model = nn.Identity()
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
            client_id=i,
            model=model,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            config=cfg.client
        )
        clients.append(client)


    # Initialize the server with a dummy model (e.g., weight vector of zeros)
    server = Server(model=base_model)
    
    metrics = {"train":
                    {"loss": [], "accuracy": []},
        "test":
                    {"loss": [], "accuracy": [], 'std accuracy': []},
    }
    
    for rnd in range(cfg.train.rounds):
        lr_update(rnd, cfg.client, clients)
        sampled_clients = random.sample(clients, k=int(cfg.train.participation_rate * len(clients)))
        train_gradient_norm = 0
        models = []
        total_train_samples = 0
        gradients = torch.Tensor(len(sampled_clients))
        for client in sampled_clients:
            n_samples, gradient_norm, gradient = client.local_train(local_epochs=cfg.train.local_epochs)
            model = client.get_state()
            for name in model:
                model[name] = model[name].cpu() * n_samples  # Scale model by number of samples
            models.append(model)
            train_gradient_norm += gradient_norm
            total_train_samples += n_samples
            
        
        train_results = {
            'gradient_norm': train_gradient_norm / max(1, len(sampled_clients))
            }
        
        log.info(f"Train, Round {rnd} : gradient_norm : {train_results['gradient_norm']}")
        
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
            
        log.info(f"Overall consensus distance : {consensus_distance:.6f}")
        
        # Server aggregates the client updates
        new_global = server.aggregate(models, np.ones(len(sampled_clients)) / total_train_samples)

        # Distribute the new global model to all clients
        for client in clients:
            client.model.load_state_dict(new_global)
            
        # Evaluate (average across clients)
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
            test_num_samples = len(client.test_loader.dataset) # type: ignore[attr-defined]
            test_weighted_loss += test_loss * test_num_samples
            test_weighted_acc += test_acc * test_num_samples
            accuracies.append(test_acc)
            total_test_samples += test_num_samples

            ###### TRAIN METRICS #######
            train_loss, train_acc = client.evaluate(client.train_loader)
            train_num_samples = len(client.train_loader.dataset) # type: ignore[attr-defined]
            train_weighted_loss += train_loss * train_num_samples
            train_weighted_acc += train_acc * train_num_samples
            total_train_samples += train_num_samples

        test_avg_loss = test_weighted_loss / total_test_samples
        test_avg_acc = test_weighted_acc / total_test_samples
        test_std_acc = np.std(accuracies)
        train_avg_loss = train_weighted_loss / total_train_samples
        train_avg_acc = train_weighted_acc / total_train_samples

        metrics['train']['loss'].append(train_avg_loss)
        metrics['train']['accuracy'].append(train_avg_acc)
        metrics['test']['loss'].append(test_avg_loss)
        metrics['test']['accuracy'].append(test_avg_acc)
        metrics['test']['std accuracy'].append(test_std_acc)

        log.info(f"Train, Round {rnd} : loss => {train_avg_loss},  accuracy: {train_avg_acc}")
        log.info(f"Test, Round {rnd} : loss => {test_avg_loss},  accuracy: {test_avg_acc}, std: {test_std_acc}")
        


    print_metrics(metrics['train'], 'Train')
    print_metrics(metrics['test'], 'Test')

if __name__ == "__main__":
    simulate_fedavg()
