#!/usr/bin/env bash
set -u

# GPLab smoke test runner.
# Runs a minimal 1-epoch experiment across built-in pools and TU datasets.
# Results are written to a TSV file for quick inspection.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

DEFAULT_POOLS=(
  nopool
  topkpool
  sagpool
  asapool
  sparsepool
  mincutpool
  diffpool
  densepool
)

DEFAULT_DATASETS=(
  MUTAG
  PROTEINS
  ENZYMES
  FRANKENSTEIN
  Mutagenicity
  AIDS
  DD
  NCI1
  COX2
)

CONFIG_PATH="${CONFIG_PATH:-/tmp/gplab_smoke_experiment.toml}"
SEEDS_PATH="${SEEDS_PATH:-/tmp/gplab_smoke_seeds}"
RESULTS_PATH="${RESULTS_PATH:-/tmp/gplab_smoke_results.tsv}"
LOG_FILE="${LOG_FILE:-}"
TAG_PREFIX="${TAG_PREFIX:-smoke}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
POOL_RATIO="${POOL_RATIO:-0.5}"
RUNS="${RUNS:-1}"
LR="${LR:-0.0005}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATIENCE="${PATIENCE:-0}"
EPOCHS="${EPOCHS:-1}"
SEED_BASE="${SEED_BASE:-20260320}"

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

cat >"$CONFIG_PATH" <<EOF
[experiment]
runs = $RUNS
lr = $LR
batch_size = $BATCH_SIZE
patience = $PATIENCE
epochs = $EPOCHS
seeds = "$SEEDS_PATH"
seed_mode = "auto"
seed_base = $SEED_BASE
allow_duplicate_seeds = false
train_ratio = 0.8
val_ratio = 0.1
EOF

printf '101\n' >"$SEEDS_PATH"

printf 'pool\tdataset\tstatus\texit_code\tseconds\n' >"$RESULTS_PATH"

if ! sh -c "$PYTHON_CMD --version" >/dev/null 2>&1; then
  echo "PYTHON_CMD is not runnable: $PYTHON_CMD" >&2
  echo "Set PYTHON_CMD to a working command, for example:" >&2
  echo "PYTHON_CMD='conda run -n torch_env python3' bash utils/smoke_test.sh" >&2
  exit 1
fi

for pool in "${POOL_LIST[@]}"; do
  for dataset in "${DATASET_LIST[@]}"; do
    start="$(date +%s)"
    cmd="$PYTHON_CMD train_cli.py --pool \"$pool\" --pool-ratio \"$POOL_RATIO\" --dataset \"$dataset\" --experiment-config \"$CONFIG_PATH\" --tag \"${TAG_PREFIX}_${pool}_${dataset}\""
    if [ -n "$LOG_FILE" ]; then
      cmd="$cmd --log-file \"$LOG_FILE\""
    fi
    sh -c "$cmd" \
      >/tmp/gplab_smoke_stdout.log 2>/tmp/gplab_smoke_stderr.log
    exit_code=$?
    end="$(date +%s)"
    elapsed="$((end - start))"

    if [ "$exit_code" -eq 0 ]; then
      status="ok"
    else
      status="$(tail -n 1 /tmp/gplab_smoke_stderr.log | tr '\t' ' ' | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
      if [ -z "$status" ]; then
        status="failed"
      fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' "$pool" "$dataset" "$status" "$exit_code" "$elapsed" >>"$RESULTS_PATH"
    printf '[%s][%s] %s (%ss)\n' "$pool" "$dataset" "$status" "$elapsed"
  done
done

printf '\nSaved smoke test results to %s\n' "$RESULTS_PATH"
