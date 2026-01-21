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
```yaml
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
See the code in `matcha_mixing_matrix.py`

## Memory Activation
In config, you can activate the memory principle so that, this formula applies in the gossiping phase : 
```math
x_i^{(t+1)} =  \sum_{j\in \mathcal{N}_i} W_{ij}b_j^{(t)} \quad \text{where} \quad b_j^{(t+1)} =  \begin{cases}
        w_j^{(t+\frac{1}{2})} \quad \text{if node $j$ participate in communication}  \\
        b_j^{(t)} \quad \text{else we take the previous model}
\end{cases}
```
To activate it, you change the line `old_gradients: true`


## Gradient Selection
In config, you can activate the gradient selection principle so only gradient veryfing :
```math
    \frac{||{\nabla F_i(w_i^{(t+1)}, \xi_i^{(t+1)}) - \nabla F_i(w_i^{(t)}, \xi_i^{(t)})||}}{||\nabla F_i(w_i^{(t)}, \xi_i^{(t)})||} \geq r
```
transmit their new computed model.
```yaml
selection_method: 
  name: gradient_based # You can set it to whatever you want not to have gradient_selection
  threshold: 6
```

## Structure of graph_runner
<strong>Main Methods</strong>
<ul>
  <li><code>`__init__(...)`</code></li>
  Initializes clients, graph topology, gossip parameters, training hyperparameters, and internal metric buffers. In "two_clusters" mode, it automatically identifies central and border nodes for cluster-level diagnostics.

  <li><code>`training(rnd)`</code></li>
  Runs local training on all clients for the given round, stores client model states, aggregates gradient statistics, and logs global gradient norms.

  <li><code>`gossip_phase(W)`</code></li>
  Performs decentralized model aggregation using the provided mixing matrix W. Each active client pulls neighbor models and updates its own state accordingly.

  <li><code>`load_metrics(rnd)`</code></li>
  Evaluates all clients on train and test sets, computes weighted averages, tracks consensus distances, and logs round-level performance statistics.

  <li><code>`run()`</code></li>
  The main execution loop. For each round, it:
  <ol>
    <li>Optionally updates learning rates.</li>
    <li>Triggers local training.</li>
    <li>Selects active edges or nodes (depending on the communication strategy).</li>
    <li>Computes or updates the mixing matrix.</li>
    <li>Runs the gossip phase.</li>
    <li>Evaluates and logs metrics.</li>
    <li>Returns a dictionary of recorded metrics.</li>
  </ol>

```
{
    "train": {"loss": [...], "accuracy": [...]},
    "test": {"loss": [...], "accuracy": [...], "std accuracy": [...]}
}
```
## Launch an experiment
To launch a new experiment, you have to specify the right configuration as mentionned above in the `Configuration` chapter.<br>
Then, you run :
```
python3 main.py
```

## Get metrics and plots
To be able to generate plots and metrics analysis, you have to go into the `compare_training.ipynb`. In this file, you find in the first box :
```python
log_path = 'negative_possibility/2025-12-03/11-56-47'
log_path2 = "negative_possibility/2025-12-01/"
log_path3 = 'Square_distance/2025-11-28/10-59-04'
match_log = 'fixed_value/2025-10-24/16-59-25'
log_name = 'fixed_value_training.log'
```
This allows to select multiple log directories for comparaison of their metrics in plots. You also have to specify the `log_name` variable which the name of the log file the code has to go look into to get the metrics. At this moment, the two possibilities are `fixed_value_training.log` and `main.log`