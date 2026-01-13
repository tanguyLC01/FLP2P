from typing import Dict, Optional, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
import logging


#ShareMode = Literal["backbone", "full"]
log = logging.getLogger(__name__)

class FLClient:
    """Federated learning client supporting personalization by sharing only the backbone.

    - If `share_mode == "backbone"`, only the model's backbone (feature extractor) is shared.
    - If `share_mode == "full"`, the entire model is shared.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
    ) -> None:
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = config.get("learning_rate", 0.01)
        self.weight_decay = config.get("weight_decay", 0.0)
        self.momentum = config.get("momentum", 0.0)
        # Store gradients of neighbors from the previous round
        self.neighbor_models: Dict[int, Dict[str, torch.Tensor]] = {}

        
        self.total_neighbors_samples = 0
        
        
    def store_neighbor_models(self, neighbor_state: Dict[int, Dict[str, torch.Tensor]]) -> None:
        """
        Store all the gradients of neighbors from the new round.
        Args:
            neighbor_state: Dict mapping neighbor IDs to their gradients dict (param_name -> tensor)
        """
        for neighbor_id, models in neighbor_state.items():
            self.neighbor_models[neighbor_id] = copy.deepcopy(models)

    def get_stochastic_gradient(self, criterion: nn.Module = None) -> Dict[str, torch.Tensor]:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        self.model.train()
        inputs, targets = next(iter(self.train_loader))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        gradient = {name: param.grad.detach().clone() for name, param in self.model.named_parameters() if param.requires_grad}
        return gradient
            
    def _optimizer(self) -> torch.optim.Optimizer:
        weights = [v for k, v in self.model.named_parameters() if "weight" in k]
        # biases = [v for k, v in self.model.named_parameters() if "bias" in k]
        return   torch.optim.Adam(
             [
                {"params": weights, "weight_decay": self.weight_decay},
            ], lr=self.learning_rate)
        
    def local_train(self, local_epochs: int = 1, criterion: Optional[nn.Module] = None) -> Tuple[float, float]:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        optimizer = self._optimizer()
        self.model.train()
        correct = 0
        total = 0
        gradient = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}

        for _ in range(local_epochs):
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                total += targets.size(0)
                for name, param in self.model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        gradient[name] += param.grad.detach().clone() * inputs.size(0)
                optimizer.step()
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()

        num_samples = len(self.train_loader.dataset)
        for name in gradient:
            gradient[name] /= num_samples
            
        # Optionally, compute average gradient norm
        avg_grad_norm = sum(grad.norm(2).item() ** 2 for grad in gradient.values()) ** 0.5
        
        return num_samples, avg_grad_norm, self.get_stochastic_gradient()

    @torch.no_grad()
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Tuple[float, float]:
        loader = data_loader or self.test_loader
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_samples = len(loader.dataset)
        correct = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
        avg_loss = total_loss / len(loader) # Same remarks as in train
        avg_acc = correct / num_samples
        test_evaluation = "Test" if loader == self.test_loader else "Training"
        log.info(f"Evaluation [{test_evaluation} Client {self.client_id}] - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        return avg_loss, avg_acc
    
    def get_gradient_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def update_state(self, neighbor_weights: List[float]) -> None:
        """
        Update the model state using the aggregated gradient.
        Args:
            aneighbor_weights: List of weights corresponding to the stored neighbor gradients.
            alpha: Learning rate to use for the update. If None, use self.learning_rate.
        """
        if not hasattr(self, "neighbor_models"): return
        if not isinstance(neighbor_weights, torch.Tensor):
            neighbor_weights = torch.tensor(neighbor_weights, dtype=torch.float32)
        aggregated = {k: torch.zeros_like(v, dtype=torch.float32).cpu() for k, v in self.model.state_dict().items()}
        for j, state in self.neighbor_models.items():
            for k in aggregated:
                aggregated[k] += neighbor_weights[j] * state[k]

        self.model.load_state_dict({k: v.to(self.device) for k, v in aggregated.items()})

    def get_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}

    # def set_state(self, state: Dict[str, torch.Tensor]) -> None:
    #     current = self.model.state_dict()
    #     mapped = {k: v.to(current[k].device).type_as(current[k]) for k, v in state.items() if k in current}
    #     current.update(mapped)
    #     self.model.load_state_dict(current)
        
        