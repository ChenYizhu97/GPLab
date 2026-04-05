# GPLab Automation Interface

This document is for automation, agents, and scripted benchmark workflows.

If you are a human user looking for normal usage, read [`README.md`](README.md) instead.

## Interface Boundary

GPLab exposes one benchmark core with two presentation modes:

- human mode: readable console output
- automation mode: JSON-only stdout

Use `--output-format json` on supported entrypoints:

- `main.py`
- `query.py`
- `replay.py`
- `validate.py`

## Train Interface

Run one experiment with structured output:

```bash
python3 main.py \
  --pool sagpool \
  --pool-ratio 0.5 \
  --dataset PROTEINS \
  --output-format json
```

Success payload shape:

```json
{
  "ok": true,
  "kind": "train_result",
  "record": {},
  "summary": {},
  "request": {}
}
```

Failure payload shape:

```json
{
  "ok": false,
  "kind": "train_error",
  "error": {
    "type": "config_error",
    "message": "..."
  }
}
```

## Job File Interface

`main.py` accepts `--job-file <path>` for one explicit experiment request.

Minimal shape:

```json
{
  "dataset": "PROTEINS",
  "pool": {
    "name": "sagpool",
    "ratio": 0.5
  },
  "model": {
    "variant": "sum"
  },
  "train": {
    "runs": 10,
    "lr": 0.0005,
    "batch_size": 32,
    "patience": 50,
    "epochs": 500,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "seed_mode": "auto",
    "seed_base": 20260320,
    "allow_duplicate_seeds": false
  },
  "log_file": "runs/bench.jsonl",
  "tag": "baseline_proteins"
}
```

Merge order:

1. TOML defaults
2. job file
3. explicit CLI overrides

Unknown fields are rejected.

## Query Interface

Flat summaries:

```bash
python3 query.py --log-file runs/bench.jsonl --output-format json
```

Grouped benchmark report:

```bash
python3 query.py --log-file runs/bench.jsonl --report --output-format json
```

The benchmark grouping boundary is:

- `spec.dataset`
- `spec.model`
- `spec.train`

It intentionally excludes:

- `spec.pool`

## Replay Interface

Structured replay metadata:

```bash
python3 replay.py \
  --log-file runs/bench.jsonl \
  --record-id <record_id> \
  --output-format json
```

Returned metadata includes:

- normalized record summary
- generated config paths
- reconstructed command
- runtime compatibility status

Compatibility status values:

- `compatible`
- `mismatch`
- `unknown`

## Validation Interface

Smoke validation:

```bash
python3 validate.py \
  --pools sagpool,diffpool \
  --datasets MUTAG,PROTEINS \
  --output-format json
```

`validate.py` is intentionally thin at the benchmark level:

- it prepares a minimal experiment config
- it runs one case at a time through `main.py --output-format json`
- it returns per-case status objects

This subprocess design is deliberate. Running multiple training cases inside one long-lived Python process triggered unstable runtime aborts in this environment, especially when mixing repeated sparse and dense jobs. The validator therefore isolates each case behind the stable train interface instead of re-embedding the training loop.

## Dense Pooling Protocol

Dense assignment-based methods in GPLab are:

- `mincutpool`
- `diffpool`
- `densepool`

Protocol details:

- input `mask` suppresses padded input nodes before dense pooling
- pooled outputs are cluster nodes, not retained input nodes
- GPLab keeps all fixed output cluster slots when converting dense pooled graphs back to sparse format
- pooled adjacency values are preserved as `edge_weight`

This is benchmark protocol, not accidental tensor plumbing.
