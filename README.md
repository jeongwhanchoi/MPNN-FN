# Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs

<img src="img/fn.png" width="600">

> **Graph where our fractal nodes are connected.**
> 1.  After partitioning the original graph into subgraphs, we aggregate the low and high-frequency information of each subgraph to create *fractal nodes*.
> 2. Then, we propagate the information to the original nodes (see our proposed $\mathsf{FN}$). 
> 3. We also support the long-distance interactions (orange dashed lines) between fractal nodes (see our proposed $\mathsf{FN}_M$).



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
## License

This project is released under the [MIT License](LICENSE).
