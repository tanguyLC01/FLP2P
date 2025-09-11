from typing import Dict, Literal, Optional, Tuple

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
        self.prev_neighbor_gradients = {}
        
    def store_neighbor_gradients(self, neighbor_gradients: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Store all the gradients of neighbors from the previous round.
        Args:
            neighbor_gradients: Dict mapping neighbor IDs to their gradients dict (param_name -> tensor)
        """
        # Deep copy to avoid reference issues
        self.prev_neighbor_gradients = {
            neighbor_id: {k: v.clone().detach().cpu() for k, v in grads.items()}
            for neighbor_id, grads in neighbor_gradients.items()
        }

    def _optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)

    def local_train(self, local_epochs: int = 1, criterion: Optional[nn.Module] = None) -> Tuple[float, float]:
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        optimizer = self._optimizer()
        self.model.train()
        total_loss = 0.0
        correct = 0
        for _ in range(local_epochs):
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                    correct += (preds == targets).sum().item()
        num_samples = len(self.train_loader.dataset)
        avg_loss = total_loss / len(self.train_loader) / local_epochs # The loss is already average on a batch, so we take the mean on the number of batches and local_epochs
        avg_acc = correct / num_samples / local_epochs
        gradient_norm = self.get_gradient_norm()
        return avg_loss, avg_acc, num_samples, gradient_norm

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
        avg_loss = total_loss / len(self.test_loader) # Same remarks as in train
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

    def update_state(self, aggregated_gradient: Dict[str, torch.Tensor], alpha: Optional[float] = None) -> None:
        """
        Update the model state using the aggregated gradient.
        Args:
            aggregated_gradient: Dict mapping parameter names to aggregated gradients (torch.Tensor).
            alpha: Learning rate to use for the update. If None, use self.learning_rate.
        """
        if alpha is None:
            alpha = self.learning_rate
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_gradient and param.requires_grad:
                    grad = aggregated_gradient[name].to(param.device).type_as(param)
                    param.data -= alpha * grad
        
    def get_gradient(self) -> Dict[str, torch.Tensor]:
        """
        Return the current gradients of the model parameters as a dict.
        Returns:
            Dict mapping parameter names to their gradients (torch.Tensor).
            Only parameters with gradients are included.
        """
        # Compute gradients from a single batch (sampled from train_loader)
        self.model.train()
        inputs, targets = next(iter(self.train_loader))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        gradients = {}
        for name, param in self.model.named_parameters():
            gradients[name] = param.grad.clone().detach().cpu()
        return gradients


    def get_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone().detach().cpu() for k, v in self.model.state_dict().items()}

    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        current = self.model.state_dict()
        mapped = {k: v.to(current[k].device).type_as(current[k]) for k, v in state.items() if k in current}
        current.update(mapped)
        self.model.load_state_dict(current) 