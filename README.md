# GPLab

GPLab is a lightweight benchmark project for graph pooling methods on graph classification tasks.

Its purpose is simple: keep the training protocol, backbone, dataset handling, and logging format consistent so different pooling methods can be compared under the same setup.

![GPLab](GPLab.png)

## What GPLab Does

GPLab focuses on four things:

- run graph classification experiments with a fixed backbone;
- swap pooling methods from the command line;
- support both built-in and custom pooling modules;
- record each experiment as a structured JSONL entry for later querying.

This makes GPLab a compact pooling benchmark harness rather than a general training framework.

## Core Design

GPLab uses one shared graph-classification backbone and one shared experiment loop.

The key design choice is the pooling protocol:

- sparse pooling methods run directly on sparse graph batches;
- dense pooling methods are adapted through a dense-to-sparse bridge so they can reuse the same downstream sparse backbone.

This keeps the post-pooling computation path aligned across methods and improves comparability.

## Project Layout

```text
GPLab/
  main.py                    # CLI entrypoint
  experiment/
    config.py                # experiment request normalization
    runner.py                # experiment orchestration
    record.py                # result/repro record assembly
  training.py                # training and evaluation loops
  querry.py                  # JSONL query tool
  model/
    Model.py                 # shared graph classifier backbone
    Classifer_Sum.py         # sum model variant
    Classifer_plain.py       # plain model variant
  layers/
    resolver.py              # conv/pool resolver and plugin loading
    functional.py            # readout and pooling helpers
    pool/
      contracts.py           # PoolOutput contract and validation
      PoolAdapter.py         # dense pooling adapter
      SAGPool.py             # customized SAG pooling
      SparsePool.py          # custom sparse pooling
  utils/
    dataset.py               # TU dataset loading and splitting
    reproducibility.py       # seeds, loaders, runtime determinism
    data.py                  # dense/sparse conversion helpers
    io.py                    # runtime metadata and experiment printing
    jsonl.py                 # JSONL I/O
    smoke_test.sh            # built-in pool x dataset smoke test
  config/
    model.toml               # model defaults
    experiment.toml          # experiment defaults
    seeds                    # optional seed file for file mode
  examples/
    custom_pool_plugin.py    # custom pooling example
    logs/                    # sample logs
```

## Installation

GPLab depends on PyTorch, PyG, and a small set of CLI/logging utilities.

Required packages:

- `torch`
- `torch-geometric`
- `torcheval`
- `typer`
- `toml`
- `rich`
- `tqdm`
- `numpy`

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torch-geometric torcheval typer toml rich tqdm numpy
```

## Quick Start

Run one experiment:

```bash
python3 main.py --pool sagpool --pool-ratio 0.5 --dataset PROTEINS
```

Run the plain model instead of the sum model:

```bash
python3 main.py --pool sagpool --pool-ratio 0.5 --dataset PROTEINS --model-type plain
```

`--model-type` is persisted in experiment logs as `model.variant`, so `sum`
and `plain` runs remain distinguishable in JSONL records and downstream queries.

Append the result to a JSONL file:

```bash
python3 main.py \
  --pool sparsepool \
  --pool-ratio 0.5 \
  --dataset PROTEINS \
  --log-file runs/bench.jsonl \
  --comment "purpose=baseline;date=2026-03-31"
```

Batch script example:

```bash
bash utils/main.sh
```

Run a built-in smoke test across all built-in pools and TU datasets:

```bash
bash utils/smoke_test.sh
```

To limit the sweep, override `POOLS` or `DATASETS`:

```bash
POOLS="sagpool diffpool" DATASETS="MUTAG PROTEINS" bash utils/smoke_test.sh
```

If your environment does not expose the correct Python by default, pass it explicitly:

```bash
PYTHON_CMD="conda run -n torch_env python3" bash utils/smoke_test.sh
```

## Querying Logs

Query a JSONL log file:

```bash
python3 querry.py --log-file runs/bench.jsonl
```

Filter by model variant:

```bash
python3 querry.py --log-file runs/bench.jsonl --model-type plain
```

`querry.py` reads `model.variant` from each record and defaults missing values to `sum`
for backward compatibility with older logs.

## Models

GPLab currently provides two model variants built on the same shared backbone:

- `sum`: reads out graph representations both before and after pooling, then adds them;
- `plain`: uses only the post-pooling graph representation.

Both models share:

- `pre_gnn`
- two graph convolution blocks
- one pooling stage
- one shared readout definition
- one post-MLP classification head

## Built-in Pooling Methods

Available built-ins:

- `nopool`
- `topkpool`
- `sagpool`
- `asapool`
- `sparsepool`
- `mincutpool`
- `diffpool`
- `densepool`

## Pooling Contract

All pooling layers are expected to return a `PoolOutput` dataclass.

Minimal required fields:

- `x`
- `edge_index`
- `batch`

Optional fields:

- `edge_attr`
- `perm`
- `score`
- `aux_loss`

The first batch is validated through `validate_pool_output()` so format errors can fail early and clearly.

Example:

```python
from layers.pool.contracts import PoolOutput


class MyPool(torch.nn.Module):
    def forward(self, x, edge_index, batch):
        return PoolOutput(
            x=x,
            edge_index=edge_index,
            batch=batch,
            edge_attr=None,
            perm=None,
            score=None,
            aux_loss=None,
        )
```

## Custom Pooling Plugin

Custom pooling is loaded with:

```text
<python_module>:<factory_name>
```

Example:

```bash
python3 main.py \
  --pool examples.custom_pool_plugin:build_pool \
  --pool-ratio 0.6 \
  --dataset PROTEINS
```

Recommended factory signature:

```python
def build_pool(
    in_channels: int,
    ratio: float = 0.5,
    avg_node_num=None,
    nonlinearity="relu",
):
    ...
```

GPLab first tries the full signature and then falls back to `(in_channels, ratio)`.

## Dense Pooling Adapter

`mincutpool`, `diffpool`, and `densepool` are integrated through `PoolAdapter`.

The adapter does four things:

1. convert sparse batches to dense tensors;
2. compute the assignment matrix;
3. apply the dense pooling method;
4. convert the pooled graph back to sparse format.

This lets dense pooling methods share the same downstream sparse backbone used by sparse pooling methods.

## Configuration

### `config/model.toml`

Main model fields:

- `hidden_features`
- `nonlinearity`
- `p_dropout`
- `conv_layer`
- `pre_gnn`
- `post_gnn`
- `variant` in logged records, set from `--model-type`

### `config/experiment.toml`

Main experiment fields:

- `runs`
- `lr`
- `batch_size`
- `patience`
- `epochs`
- `seeds`
- `seed_mode`
- `seed_base`
- `allow_duplicate_seeds`
- `train_ratio`
- `val_ratio`

`test_ratio` is derived as:

```text
1 - train_ratio - val_ratio
```

## Reproducibility

GPLab tracks reproducibility at multiple levels:

- seeded Python / NumPy / Torch randomness;
- seeded DataLoader generator;
- deterministic split fingerprint via `split_digest`;
- runtime environment metadata;
- full experiment configuration recorded in each JSONL entry.

Current seed modes:

- `auto`: generate deterministic unique seeds from `seed_base`;
- `file`: load seeds from file, with duplicate protection unless explicitly allowed.

## Querying Results

Use `querry.py` to inspect JSONL logs.

Examples:

```bash
python3 querry.py --log-file runs/bench.jsonl --pool sparsepool --dataset PROTEINS --epoch
```

```bash
python3 querry.py --log-file runs/bench.jsonl --show-repro
```

```bash
python3 querry.py --log-file runs/bench.jsonl --verify-repro
```

## Logged Record Structure

Each experiment record contains the main sections:

- `model`
- `experiment`
- `pool`
- `dataset`
- `comment`
- `meta`
- `repro`
- `results`

The `results` block contains:

- aggregate statistics;
- arrays of losses / accuracies / stop epochs;
- per-run detailed records.

## Current Scope

GPLab is intentionally narrow in scope:

- TU datasets only;
- graph classification only;
- one shared backbone family;
- one pooling comparison protocol.

This constraint is deliberate. The project is optimized for controlled pooling comparison, not for broad task coverage.
