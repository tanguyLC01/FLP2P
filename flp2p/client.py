from typing import Dict, Literal, Optional, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader

#ShareMode = Literal["backbone", "full"]


class FLClient:
    """Federated learning client supporting personalization by sharing only the backbone.

    - If `share_mode == "backbone"`, only the model's backbone (feature extractor) is shared.
    - If `share_mode == "full"`, the entire model is shared.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = config.get("learning_rate", 0.01)
        self.weight_decay = config.get("weight_decay", 0.0)
        self.momentum = config.get("momentum", 0.0)
        # Store gradients of neighbors from the previous round
        self.neighbor_gradients: Dict[int, Dict[str, torch.Tensor]] = {}
        
        self.total_neighbors_samples = 0
        
        
    def store_neighbor_gradients(self, neighbor_gradients: Dict[int, Dict[str, torch.Tensor]]) -> None:
        """
        Store all the gradients of neighbors from the new round.
        Args:
            neighbor_gradients: Dict mapping neighbor IDs to their gradients dict (param_name -> tensor)
        """
        for neighbor_id, gradients in neighbor_gradients.items():
            self.neighbor_gradients[neighbor_id] = gradients.copy()
        del neighbor_gradients
            
    def _optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

    def local_train(self, local_epochs: int = 1, criterion: Optional[nn.Module] = None) -> Tuple[float, float]:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        optimizer = self._optimizer()
        self.model.train()
        correct = 0
        total = 0
        gradient = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
        batch_count = 0

        for _ in range(local_epochs):
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total += targets.size(0)
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                batch_count += 1

        num_samples = len(self.train_loader.dataset)
        
        # Compute average gradient
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                gradient[name] = param.grad.detach().clone()
        # Optionally, compute average gradient norm
        avg_grad_norm = sum(grad.norm(2).item() ** 2 for grad in gradient.values()) ** 0.5
        
        return num_samples, avg_grad_norm, gradient

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
        return avg_loss, avg_acc
    
    def get_gradient_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def aggregate_gradients_weighted(self, gradients_dict:  Dict[int, Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients weighted by the adjacency matrix (weights).
        Args:
            gradients_list: List of gradients dicts (param_name -> tensor) from neighbors.
            weights: List of weights (same order as gradients_list) from adjacency matrix.
        Returns:
            Dict of aggregated gradients (param_name -> tensor).
        """
        #if not gradients_dict or not weights or len(gradients_dict) != len(weights):
        #    return {}
        agg: Dict[str, torch.Tensor] = {}
        param_names = list(gradients_dict.values())[0].keys()
        for name in param_names:
            weighted_grads = []
            for idx in gradients_dict:
                weighted_grads.append(gradients_dict[idx][name].float() * weights[idx])
            if weighted_grads:
                agg[name] = sum(weighted_grads)
        return agg

    def update_state(self, neighbor_weights: List[float], consensus_lr: int = 0.1) -> None:
        """
        Update the model state using the aggregated gradient.
        Args:
            aneighbor_weights: List of weights corresponding to the stored neighbor gradients.
            alpha: Learning rate to use for the update. If None, use self.learning_rate.
        """
        if not self.neighbor_gradients:
            return
        
        aggregated_gradient = self.aggregate_gradients_weighted(
            gradients_dict=self.neighbor_gradients,
            weights=neighbor_weights)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_gradient and param.requires_grad:
                    grad = aggregated_gradient[name].to(param.device).type_as(param)
                    param.data -= consensus_lr * grad
        
    def get_gradient(self) -> Dict[str, torch.Tensor]:
        """
        Return the current gradients of the model parameters as a dict.
        Returns:
            Dict mapping parameter names to their gradients (torch.Tensor).
            Only parameters with gradients are included.
        """
        # Compute gradients from a single batch (sampled from train_loader)
        gradients = {}
        for name, param in self.model.named_parameters():
            gradients[name] = param.grad.clone().detach().cpu()
        return gradients

    def get_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        current = self.model.state_dict()
        mapped = {k: v.to(current[k].device).type_as(current[k]) for k, v in state.items() if k in current}
        current.update(mapped)
        self.model.load_state_dict(current)