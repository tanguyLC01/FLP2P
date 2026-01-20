# Project Overview
The goal of this project is to create a Distributed Federated Learning framework to be able to run decentralized algorithms with PyTorch models.

## Setup

```bash
python3 -m venv ./personalisation_env
source ./personalisation_env/bin/activate
pip install -r requirements.txt
```

## 

## Configuration
The configuration files are in the `conf` folder. We use Hydra system to be able to handle multiple configurations at once.<br>
### Graph
You find all the possible graphs you can generate in this repo and their specific parameter. The repo is mainly built (metrics analysis and plots) for the two cluster topology.

### config.yaml
We present a generic config file to run an a DFL experiment.
```
defaults:
  - data: cifar10
  - model: lenet5
  - partition: dirichlet
  - client: default
  - train: default
  - graph: two_clusters
  - _self_
  - override hydra/launcher: joblib



seed: 42
use_cuda: true
mixing_matrix: jaccard
run_name: small_graph
old_gradients: false
aggregation_step_per_round: 1
hydra:
  run:
    dir: ./${run_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: ./${run_name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${mixing_matrix}-selection_method:${selection_method.threshold}-old_gradients:${old_gradients}
same_distrib_test_set: false
selection_method: 
  name: normal
  threshold: 0
```
In this config, we run LeNet5 on every client who all have a fraction of the CIFAR-10 dataset, partitioned along the dirichlet distribution.
We set the mixing/gossiping matrix to Jaccard. The available matrices are "jaccard", "metroplis_hasting", "maximum_degree" and the "matcha" method.
#### Metroplis Hasting
```math
\left[ W \right]_{ij} = 
\begin{cases} 
    \frac{1}{1 + \max\{d_i(t), d_j(t)\}} & \{i, j\} \in \mathcal{E} \\
    1 - \sum_{k \in \mathcal{N}_i(t)} W_{ik}(t) & i = j \\
    0 & \text{otherwise}
\end{cases}
```

#### Jaccard
```math
\left[ W \right]_{ij} = 
\begin{cases} 
    \frac{|\mathcal{N}_i \cap \mathcal{N}_j|}{|\mathcal{N}_i \cup \mathcal{N}_j|}  & \{i, j\} \in \mathcal{E} \\
    1 - \sum_{k \in \mathcal{N}_i(t)} W_{ik}(t) & i = j \\
    0 & \text{otherwise}
\end{cases}
```

#### Maximum Degree
```math
\left[ W \right]_{ij} = \begin{cases} 
    \frac{1}{1+\max_{i \in \mathcal{V}}d_i} \quad &\text{for} \quad (i,j) \in \mathcal{E} \quad \text{or} \quad i=j \\
    0 &\text{otherwise}
\end{cases}
```

#### Matcba
see the code in `matcha_mixing_matrix.py`




