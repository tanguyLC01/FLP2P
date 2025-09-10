import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def get_mnist_datasets(root: str = "./data") -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return train, test

def get_cifar10_datasets(root: str = "./data") -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    return train, test


def iid_partition(num_clients: int, num_samples: int) -> List[np.ndarray]:
    indices = np.random.permutation(num_samples)
    splits = np.array_split(indices, num_clients)
    return [np.array(split, dtype=np.int64) for split in splits]


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float) -> List[np.ndarray]:
    num_classes = labels.max() + 1
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for y in range(num_classes):
        np.random.shuffle(class_indices[y])
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        splits = (np.cumsum(proportions) * len(class_indices[y])).astype(int)[:-1]
        shards = np.split(class_indices[y], splits)
        for client_id, shard in enumerate(shards):
            client_indices[client_id].extend(shard.tolist())

    return [np.array(sorted(idxs), dtype=np.int64) for idxs in client_indices]

def plot_partition_distribution(partitions: List[np.ndarray], labels: np.ndarray, title: str = "Data Distribution", path_to_save: str = "./") -> None:
    num_clients = len(partitions)
    num_classes = labels.max() + 1
    class_counts = np.zeros((num_clients, num_classes), dtype=int)
    for client_id, idxs in enumerate(partitions):
        client_labels = labels[idxs]
        for c in range(num_classes):
            class_counts[client_id, c] = np.sum(client_labels == c)

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(num_clients)
    for c in range(num_classes):
        plt.bar(
            np.arange(num_clients),
            class_counts[:, c],
            bottom=bottom,
            label=f'Class {c}'
        )
        bottom += class_counts[:, c]
    plt.xlabel('Client')
    plt.ylabel('Number of samples')
    plt.title('Number of samples per client (stacked by class)')
    plt.xticks(np.arange(num_clients))
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_save, "client_class_repartition.png"))
    plt.close()

def get_dataset(config: Dict) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if config.name == "mnist":
        return get_mnist_datasets(root=config.root)
    elif config.name == "cifar10":
        return get_cifar10_datasets(root=config.root)
    else:
        raise ValueError(f"Unknown dataset: {config.name}")

def build_client_loaders(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    config: Dict,
    save_plot_path: str = "./"
) -> List[Tuple[DataLoader, DataLoader]]:
    labels = np.array(train_dataset.targets)
    if config.partition.strategy == "iid":
        parts = iid_partition(num_clients=config.partition.num_clients, num_samples=len(train_dataset))
    elif config.partition.strategy == "dirichlet":
        parts = dirichlet_partition(labels=labels, num_clients=config.partition.num_clients, alpha=config.partition.dirichlet_alpha)
    else:
        raise ValueError(f"Unknown partition strategy: {config.partition.strategy}")

    plot_partition_distribution(parts, labels, path_to_save=save_plot_path)
    loaders: List[Tuple[DataLoader, DataLoader]] = []
    for idxs in parts:
        train_subset = Subset(train_dataset, indices=idxs.tolist())
        # Use the full test set for all clients by default
        train_loader = DataLoader(train_subset, batch_size=config.get("batch_size", 64), shuffle=True, num_workers=config.get("num_workers", 2))
        test_loader = DataLoader(test_dataset, batch_size=config.get("batch_size", 64), shuffle=False, num_workers=config.get("num_workers", 2))
        loaders.append((train_loader, test_loader))
    return loaders 