#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AIRR_PATH="${AIRR_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
LOCUS="${LOCUS:-}"
TARGET_COL="${TARGET_COL:-log10_pgen_1mm}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_LEN="${MAX_LEN:-40}"
CLONE_ID_COL="${CLONE_ID_COL:-clone_id}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DROPOUT="${DROPOUT:-0.2}"
ENCODER_TYPE="${ENCODER_TYPE:-plain_conv}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
MLP_DIM="${MLP_DIM:-512}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-1024}"
DILATIONS="${DILATIONS:-1,2,4,8}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
SCHEDULER="${SCHEDULER:-plateau}"
SCHEDULER_PATIENCE="${SCHEDULER_PATIENCE:-2}"
SCHEDULER_FACTOR="${SCHEDULER_FACTOR:-0.5}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "$AIRR_PATH" ]]; then
  echo "AIRR_PATH is required" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR is required" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cmd=(
  "$PYTHON_BIN" -m irrm_codec.train_pgen
  --airr-path "$AIRR_PATH"
  --output-dir "$OUTPUT_DIR"
  --target-col "$TARGET_COL"
  --clone-id-col "$CLONE_ID_COL"
  --max-len "$MAX_LEN"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --dropout "$DROPOUT"
  --encoder-type "$ENCODER_TYPE"
  --hidden-dim "$HIDDEN_DIM"
  --mlp-dim "$MLP_DIM"
  --mlp-hidden-dim "$MLP_HIDDEN_DIM"
  --dilations "$DILATIONS"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --scheduler "$SCHEDULER"
  --scheduler-patience "$SCHEDULER_PATIENCE"
  --scheduler-factor "$SCHEDULER_FACTOR"
  --train-fraction "$TRAIN_FRACTION"
  --val-fraction "$VAL_FRACTION"
  --seed "$SEED"
  --num-workers "$NUM_WORKERS"
  --log-interval "$LOG_INTERVAL"
  --no-progress
)

if [[ -n "$LOCUS" ]]; then
  cmd+=(--locus "$LOCUS")
fi

echo "root_dir=$ROOT_DIR"
echo "airr_path=$AIRR_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "locus=${LOCUS:-<none>}"
echo "target_col=$TARGET_COL"
echo "epochs=$EPOCHS batch_size=$BATCH_SIZE num_workers=$NUM_WORKERS"

"${cmd[@]}"
