from typing import Tuple, Dict
from collections import OrderedDict
import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, model_config: Dict) -> None:
        super(LeNet5, self).__init__()
        num_classes = model_config.get('num_classes', 10)
        in_channels = model_config.get('in_channels', 1)
        if model_config.batch_norm is True:
            self.net = nn.Sequential(
                OrderedDict(
                    [
                    ('conv1', nn.Sequential(
                        nn.Conv2d(in_channels, 6, kernel_size=5),
                        nn.BatchNorm2d(6),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2)
                    )),
                    ('conv2', nn.Sequential(
                        nn.Conv2d(6, 16, kernel_size=5),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2)
                    )),
                    ('flatten', nn.Flatten()),
                    ('fc1', nn.Sequential(
                        nn.Linear(model_config['latent_dimension'], 120),
                        nn.ReLU()
                    )),
                    ('fc2', nn.Sequential(
                        nn.Linear(120, 84),
                        nn.ReLU()
                    )),
                    ('fc3', nn.Linear(84, num_classes))
                    ]
                )
            )
        else:
            self.net = nn.Sequential(
                OrderedDict(
                    [
                    ('conv1', nn.Sequential(
                        nn.Conv2d(in_channels, 6, kernel_size=5),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2)
                    )),
                    ('conv2', nn.Sequential(
                        nn.Conv2d(6, 16, kernel_size=5),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2)
                    )),
                    ('flatten', nn.Flatten()),
                    ('fc1', nn.Sequential(
                        nn.Linear(model_config['latent_dimension'], 120),
                        nn.ReLU()
                    )),
                    ('fc2', nn.Sequential(
                        nn.Linear(120, 84),
                        nn.ReLU()
                    )),
                    ('fc3', nn.Linear(84, num_classes))
                    ]
                )
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.net(x)
        return x
