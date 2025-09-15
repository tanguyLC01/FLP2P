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


def iid_partition(num_clients: int, labels: np.ndarray) -> List[np.ndarray]:
    indices = np.random.permutation(len(labels))
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

def match_test_partition(
    train_parts: List[np.ndarray],
    train_labels: np.ndarray,
    test_labels: np.ndarray
) -> List[np.ndarray]:
    """
    Ensure test partition follows the same class distribution as train partition.
    """
    
    num_clients = len(train_parts)
    test_indices = np.arange(len(test_labels))
    test_parts: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in sorted(np.unique(train_labels)):
        train_counts = np.array([np.sum(train_labels[part] == cls) for part in train_parts])
        if train_counts.sum() == 0:
            continue

        proportions = train_counts / train_counts.sum()
        cls_indices = test_indices[test_labels == cls]
        np.random.shuffle(cls_indices)

        splits = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
        shards = np.split(cls_indices, splits)

        for client_id, shard in enumerate(shards):
            test_parts[client_id].extend(shard.tolist())

    return [np.array(sorted(idxs), dtype=np.int64) for idxs in test_parts]

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
    plt.savefig(os.path.join(path_to_save, f"{title}.png"))
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
        train_parts = iid_partition(num_clients=config.partition.num_clients, labels=labels)
        test_parts = match_test_partition(train_parts, labels, np.array(test_dataset.targets))
        
    elif config.partition.strategy == "dirichlet":
        train_parts = dirichlet_partition(labels=labels, num_clients=config.partition.num_clients, alpha=config.partition.dirichlet_alpha)
        test_parts = dirichlet_partition(labels=np.array(test_dataset.targets), num_clients=config.partition.num_clients, alpha=config.partition.dirichlet_alpha)
    else:
        raise ValueError(f"Unknown partition strategy: {config.partition.strategy}")

    plot_partition_distribution(train_parts, labels, path_to_save=save_plot_path, title="train_data_distribution")
    plot_partition_distribution(test_parts, np.array(test_dataset.targets), path_to_save=save_plot_path, title="test_data_distribution")
    
    loaders: List[Tuple[DataLoader, DataLoader]] = []
    for idxs_train, idxs_test in zip(train_parts, test_parts):
        train_subset = Subset(train_dataset, indices=idxs_train.tolist())
        test_subset = Subset(test_dataset, indices=idxs_test.tolist())
        
        # Use the full test set for all clients by default
        train_loader = DataLoader(train_subset, batch_size=config.get("batch_size", 64), shuffle=True, num_workers=config.get("num_workers", 2))
        test_loader = DataLoader(test_subset, batch_size=config.get("batch_size", 64), shuffle=False, num_workers=config.get("num_workers", 2))
        loaders.append((train_loader, test_loader))
    return loaders 