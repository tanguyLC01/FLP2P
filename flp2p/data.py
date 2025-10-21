import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import defaultdict
from joblib.externals.loky.backend.context import get_context

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

def get_fashion_mnist_datasets(root: str = "./data") -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Fashion-MNIST is grayscale (1 channel)
    ])
    train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    return train, test

def iid_partition(num_clients: int, labels: np.ndarray) -> List[np.ndarray]:
    indices = np.random.permutation(len(labels))
    splits = np.array_split(indices, num_clients)
    return [np.array(split, dtype=np.int64) for split in splits]



def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, min_partition_size: int = 10) -> List[np.ndarray]:
    num_classes = labels.max() + 1
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for y in range(num_classes):
        np.random.shuffle(class_indices[y])
        class_size = len(class_indices[y])

        # Keep sampling proportions until all partitions >= min_partition_size (if possible)
        trial = 0
        while True:
            proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
            splits = (np.cumsum(proportions) * class_size).astype(int)[:-1]
            shards = np.split(class_indices[y], splits)

            # Check if all shards are large enough
            if all(len(shard) >= min_partition_size or len(shard) == 0 for shard in shards):
                break  # valid split found
        
            if trial == 10:
                 raise ValueError(
                    "The max number of attempts (10) was reached. "
                    "Please update the values of alpha and try again."
                    )
            trial += 1
        # Assign shards to clients
        for client_id, shard in enumerate(shards):
            client_indices[client_id].extend(shard.tolist())

    return [np.array(sorted(idxs), dtype=np.int64) for idxs in client_indices]

def pathology_partition(labels: np.ndarray, num_clients: int, num_classes_per_client: int) -> List[np.ndarray]:
    

    num_classes = len(np.unique(labels))
    class_to_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)
        
    shard_per_class = num_classes_per_client * num_clients // num_classes
    assert (num_clients * num_classes_per_client) % num_classes == 0, \
    "Incompatible settings: num_clients * num_classes_per_client must be divisible by num_classes"
    
    
    dict_users = defaultdict(list)
    for label in class_to_indices.keys():
        x = np.array(class_to_indices[label])
        shards = np.array_split(x, shard_per_class)
        class_to_indices[label] = [shard.tolist() for shard in shards]
        
    rand_set_all = list(range(num_classes)) * shard_per_class
    np.random.shuffle(rand_set_all)
    rand_set_all = np.array(rand_set_all).reshape((num_clients, num_classes_per_client))

    for cid in range(num_clients):
        rand_set_label = rand_set_all[cid]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(class_to_indices[label]), replace=False)
            rand_set.append(class_to_indices[label].pop(idx))
        dict_users[cid] = np.concatenate(rand_set)
    
    return [np.array(idxs, dtype=np.int64) for idxs in dict_users.values()]


def match_test_partition(
    train_parts: List[np.ndarray],
    train_labels: np.ndarray,
    test_labels: np.ndarray
) -> List[np.ndarray]:
    """
    Ensure test partition follows the same class distribution as train partition.
    We get the proportion of each class in each client dataset.
    Then, we split the classes indices based on theses proportions.
    Lastly, we assign the splits/shards to each client.
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
    elif config.name == "fashion_mnist":
        return get_fashion_mnist_datasets(root=config.root)
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
    elif config.partition.strategy == "dirichlet":
        train_parts = dirichlet_partition(labels=labels, num_clients=config.partition.num_clients, alpha=config.partition.dirichlet_alpha, min_partition_size=config.partition.min_partition_size)
    elif config.partition.strategy == "pathological":
        train_parts = pathology_partition(labels=labels, num_clients=config.partition.num_clients, num_classes_per_client=config.partition.num_classes_per_client)
    elif config.partition.strategy == 'unbalanced_cluster':
        first_part, second_part = list(map(lambda x: np.array([train_dataset.targets[i] for i in x.tolist()]), dirichlet_partition(labels=labels, num_clients=2, alpha=config.partition.alpha_two_set)))
        train_parts = dirichlet_partition(labels=first_part, num_clients=config.partition.num_clients//2, alpha=config.partition.intra_cluster_alpha, min_partition_size=config.partition.min_partition_size)
        train_parts += dirichlet_partition(labels=second_part, num_clients=config.partition.num_clients//2, alpha=config.partition.intra_cluster_alpha, min_partition_size=config.partition.min_partition_size)
    else:
        raise ValueError(f"Unknown partition strategy: {config.partition.strategy}")

    if config.same_distrib_test_set is True:
        test_parts = match_test_partition(train_parts, train_labels=labels, test_labels=np.array(test_dataset.targets))
    else:
        test_parts =  [np.arange(len(test_dataset.targets)) for _ in range(config.partition.num_clients)]
    plot_partition_distribution(train_parts, labels, path_to_save=save_plot_path, title="train_data_distribution")
    plot_partition_distribution(test_parts, np.array(test_dataset.targets), path_to_save=save_plot_path, title="test_data_distribution")
    
    loaders: List[Tuple[DataLoader, DataLoader]] = []
    for idxs_train, idxs_test in zip(train_parts, test_parts):
        train_subset = Subset(train_dataset, indices=idxs_train.tolist())
        test_subset = Subset(test_dataset, indices=idxs_test.tolist())
        
        # Use the full test set for all clients by default
        ####### WARNING : if using Joblib, it is not possible to set num_workers > 0 see : github repo and issue
        train_loader = DataLoader(train_subset, batch_size=min(len(train_subset), config.get("batch_size", 64)), shuffle=True, num_workers=config.data.get("num_workers", 2))
        test_loader = DataLoader(test_subset, batch_size=min(len(test_subset), config.get("batch_size", 64)), shuffle=False, num_workers=config.data.get("num_workers", 2))
        loaders.append((train_loader, test_loader))
    return loaders 