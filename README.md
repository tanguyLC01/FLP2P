# FLP2P: Decentralized Federated Learning (MNIST, LeNet-5)

Minimal decentralized FL with personalization:
- Networks: `LeNet5` with separable backbone and classifier
- Clients: local train/eval, share backbone or full model
- Graph runner: gossip-style averaging over ring or Erdos-Rényi graph
- Hydra configs for easy experiment management

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
```

## Run (Hydra)

Base run using defaults (`conf/config.yaml`):

```bash
python main.py
```

Override number of clients and rounds:

```bash
python main.py partition.num_clients=3 train.rounds=2
```

Switch to Dirichlet non-IID partitioning and change alpha:

```bash
python main.py partition=@partition/dirichlet partition.alpha=0.3
```

Use Erdős-Rényi topology with p=0.3:

```bash
python main.py graph=@graph/erdos_renyi graph.er_p=0.3
```

Share the full model instead of just the backbone:

```bash
python main.py client.share_mode=full
```

Change batch size and workers:

```bash
python main.py data.batch_size=128 data.num_workers=4
```
