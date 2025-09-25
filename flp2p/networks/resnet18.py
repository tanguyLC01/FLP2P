import torch
from torchvision import models
from typing import Dict
import torch.nn as nn

def make_resnet18(model_config: Dict) -> torch.nn.Module:
    """
    Returns a ResNet-18 model.

    Args:
        num_classes: number of output classes (replaces final fc).
        pretrained: whether to load ImageNet pretrained weights.
        device: device string or torch.device.

    Returns:
        model on requested device.
    """
    model = models.resnet18(pretrained=model_config.pretrained)
    
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # remove maxpool

    # Replace final fully-connected to match num_classes
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, model_config.num_classes)

    return model