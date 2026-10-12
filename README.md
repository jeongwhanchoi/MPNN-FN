# Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs

[![AAAI 2026 Oral](https://img.shields.io/badge/AAAI%202026-Oral-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/39191)
[![arXiv](https://img.shields.io/badge/arXiv-2511.13010-b31b1b.svg)](https://arxiv.org/abs/2511.13010)

Official PyTorch implementation of **"Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs"** (AAAI 2026, Oral).

[Jeongwhan Choi](https://jeongwhanchoi.com), Seungjun Park, Sumin Park, Sung-Bae Cho, Noseong Park

<img src="img/fn.png" width="600">

> **Graph where our fractal nodes are connected.**
> 1.  After partitioning the original graph into subgraphs, we aggregate the low and high-frequency information of each subgraph to create *fractal nodes*.
> 2. Then, we propagate the information to the original nodes (see our proposed $\mathsf{FN}$). 
> 3. We also support the long-distance interactions (orange dashed lines) between fractal nodes (see our proposed $\mathsf{FN}_M$).

## TL;DR

Fractal nodes are a plug-in for any MPNN. Each fractal node summarizes one METIS subgraph and coexists with the original nodes, which gives shortcut paths for long-range messages and reduces over-squashing. MPNNs with fractal nodes match or beat graph Transformers on graph-level benchmarks while keeping the linear cost of message passing.

## Setup

```bash
conda create --name graph_mlpmixer python=3.8
conda activate graph_mlpmixer

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c conda-forge rdkit
pip install yacs
pip install tensorboard
pip install networkx
pip install einops

conda install -c conda-forge metis
```

## Repository Structure

```
core/
  model.py               # GraphFractal (FN) and GraphFractalMixer (FN_M)
  config.py              # default hyperparameters (yacs)
  transform.py           # METIS partitioning and positional encodings
  model_utils/           # MPNN layers (GCN, GINE, GatedGCN, ...) and MLP-Mixer
  data_utils/            # Peptides, SR25, CSL loaders
train/
  configs/GraphFractal/       # configs for FN
  configs/GraphFractalMixer/  # configs for FN_M
  <dataset>.py                # training entry points
```

Datasets are downloaded to `dataset/` on first run, and logs are written to `results/`.

## How to Run

Each script loads `train/configs/GraphFractal/<dataset>.yaml` by default. Any option in [core/config.py](core/config.py) can be overridden on the command line as `key value` pairs. To run $\mathsf{FN}_M$ with its own config, pass `--config`:

```bash
python -m train.peptides_func --config train/configs/GraphFractalMixer/peptides_func.yaml
```

Supported datasets: `peptides_func`, `peptides_struct`, `molhiv`, `moltox21`, `csl`, `sr25`.

Key options:

| Option | Values | Description |
|---|---|---|
| `model.gnn_type` | `GCNConv`, `GINEConv`, `ResGatedGraphConv`, ... | Base MPNN |
| `model.gMHA_type` | `None` / `MLPMixer` | `None` gives $\mathsf{FN}$, `MLPMixer` gives $\mathsf{FN}_M$ |
| `fractal.alpha_type` | `sc` / `vc` | Scalar or vector (per-subgraph) coefficient for the high-frequency component |
| `fractal.hpf` | `True` / `False` | Use the high-frequency component in fractal nodes |
| `metis.n_patches` | int | Number of subgraphs (= number of fractal nodes) |

- Examples:
```bash
# GCN + FN
python -m train.peptides_struct device 0 fractal.alpha_type vc metis.n_patches 32 model.gnn_type GCNConv fractal.hpf True

# GCN + FN_M
python -m train.peptides_func device 0 fractal.alpha_type vc metis.n_patches 32 model.gMHA_type MLPMixer model.gnn_type GCNConv fractal.hpf True
```

- Best hyperparameters for Peptides-struct:

```bash
# GatedGCN + FN_M
python -m train.peptides_struct device 0 fractal.alpha_type sc metis.n_patches 32 model.gMHA_type MLPMixer model.gnn_type ResGatedGraphConv fractal.hpf True model.nlayer_gnn 4 model.nlayer_mlpmixer 2  model.hidden_size 128 fractal.alpha_init zero metis.num_hops 1 metis.drop_rate 0.2
```
```

## Related Work from Our Group

Our other work on message passing, over-smoothing, and over-squashing in GNNs:

| Paper | Venue | Topic | Links |
|---|---|---|---|
| PANDA: Expanded Width-Aware Message Passing Beyond Rewiring | ICML 2024 | Over-squashing, width expansion for high-centrality nodes | [paper](https://arxiv.org/abs/2406.03671) · [code](https://github.com/jeongwhanchoi/PANDA) |
| GREAD: Graph Neural Reaction-Diffusion Networks | ICML 2023 | Over-smoothing and heterophily, reaction-diffusion graph ODEs | [paper](https://arxiv.org/abs/2211.14208) · [code](https://github.com/jeongwhanchoi/GREAD) |
| Graph Convolutions Enrich the Self-Attention in Transformers! | NeurIPS 2024 | Over-smoothing in Transformers, graph-filter-based self-attention | [paper](https://arxiv.org/abs/2312.04234) · [code](https://github.com/jeongwhanchoi/GFSA) |
| PIORF: Physics-Informed Ollivier-Ricci Flow for Long-Range Interactions in Mesh Graph Neural Networks | ICLR 2025 | Over-squashing in mesh GNNs, curvature-based rewiring | [paper](https://arxiv.org/abs/2504.04052) |
| Learning Posterior Predictive Distributions for Node Classification from Synthetic Graph Priors | ICLR 2026 | Graph prior-data fitted network (NodePFN) | [paper](https://openreview.net/forum?id=FmxRzlu0rT) · [code](https://github.com/jeongwhanchoi/NodePFN) |

## License

This project is released under the [MIT License](LICENSE).
