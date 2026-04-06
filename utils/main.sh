#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_CMD="${PYTHON_CMD:-/home/yizhu/miniconda3/bin/conda run -n torch_env python}"
LOG_FILE="${LOG_FILE:-runs/batch.jsonl}"
TAG="${TAG:-batch_run}"
POOL_RATIO="${POOL_RATIO:-0.5}"
MODEL_TYPE="${MODEL_TYPE:-sum}"

DEFAULT_POOLS=(
  sparsepool
)

DEFAULT_DATASETS=(
  PROTEINS
  ENZYMES
  Mutagenicity
  DD
  NCI1
  COX2
)

if [ -n "${POOLS:-}" ]; then
  # shellcheck disable=SC2206
  POOL_LIST=(${POOLS})
else
  POOL_LIST=("${DEFAULT_POOLS[@]}")
fi

if [ -n "${DATASETS:-}" ]; then
  # shellcheck disable=SC2206
  DATASET_LIST=(${DATASETS})
else
  DATASET_LIST=("${DEFAULT_DATASETS[@]}")
fi

if ! sh -c "$PYTHON_CMD --version" >/dev/null 2>&1; then
  echo "PYTHON_CMD is not runnable: $PYTHON_CMD" >&2
  exit 1
fi

for dataset in "${DATASET_LIST[@]}"; do
  for pool in "${POOL_LIST[@]}"; do
    echo "[run] dataset=$dataset pool=$pool model=$MODEL_TYPE tag=${TAG}_${pool}_${dataset}"
    sh -c "$PYTHON_CMD train_cli.py --pool \"$pool\" --pool-ratio \"$POOL_RATIO\" --dataset \"$dataset\" --model-type \"$MODEL_TYPE\" --log-file \"$LOG_FILE\" --tag \"${TAG}_${pool}_${dataset}\""
  done
done
