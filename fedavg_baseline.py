import numpy as np
import random
import flp2p  # Assuming the flp2p module is installed and configured
from flp2p.client import Client  # Import Client from flp2p/client.py
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from flp2p.data import build_client_loaders, get_dataset
from typing import List
from flp2p.networks.lenet5 import LeNet5
from flp2p.networks.resnet18 import make_resnet18
from flp2p.client import FLClient
from logging import log
"""
This file implements a simplified classical FedAvg simulation
using the flp2p framework style. In this example, each client
performs a dummy local update, and the server aggregates the
local models by averaging their weights.
"""

# Define a Server class that handles aggregation
class Server:
    def __init__(self, model):
        # Initialize a dummy global model as a weight vector of given size
        self.model = model

    def aggregate(self, gradients, weights):
        """
        Aggregate client models with weighted averaging.
        For simplicity, we use a simple average.
        """
        # Simple average of client gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = np.average([gradients[i][name].data * weights[i] for i in range(len(gradients))], axis=0)
        return self.model.state_dict()

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
          
    log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
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

    


    log.info(f"Train, Round {rnd} : gradient_norm : {train_results['gradient_norm']}")       

    # Initialize the server with a dummy model (e.g., weight vector of zeros)
    server = Server(model=base_model)

    # Begin the federated learning simulation
    flp2p.log("Starting FedAvg simulation")
    
    metrics = {"train":
                    {"loss": [], "accuracy": []},
        "test":
                    {"loss": [], "accuracy": [], 'std accuracy': []},
    }
    
    for rnd in range(cfg.rounds):

        sampled_clients = np.random.choice(clients, size=int(cfg.train.participation_rate * len(clients)), replace=False)
        train_gradient_norm = 0
        accumulated_gradients = []
        total_train_samples = 0
        for client in sampled_clients:
            n_samples, gradient_norm, gradients = client.local_train(local_epochs=cfg.train.local_epochs)
            for name, param in client.model.named_parameters():
                if param.requires_grad:
                    gradients[name] *= n_samples / cfg.data.batch_size  
            accumulated_gradients.append(gradients) 
            train_gradient_norm += gradient_norm
            total_train_samples += n_samples
            
        train_results = {
            'gradient_norm': train_gradient_norm / max(1, len(sampled_clients))
            }
        
        log.info(f"Train, Round {rnd} : gradient_norm : {train_results['gradient_norm']}") 
        # Server aggregates the client updates
        new_global = server.aggregate(accumulated_gradients, np.ones(len(sampled_clients)) / total_train_samples)

        # Distribute the new global model to all clients
        for client in clients:
            client.model.load_state_dict(new_global)

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


    flp2p.log("FedAvg simulation completed.")

if __name__ == "__main__":
    simulate_fedavg()
