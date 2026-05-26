#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

AIRR_PATH="${AIRR_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
LOCUS="${LOCUS:-}"
CHAIN="${CHAIN:-TRB}"
SPECIES="${SPECIES:-human}"
MODEL_PATH="${MODEL_PATH:-}"
MIRPY_PATH="${MIRPY_PATH:-}"
CLONE_ID_COL="${CLONE_ID_COL:-clone_id}"
CDR3_COL="${CDR3_COL:-junction_aa}"
THREADS="${THREADS:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
PGEN_COL="${PGEN_COL:-pgen_1mm}"
LOG10_PGEN_COL="${LOG10_PGEN_COL:-log10_pgen_1mm}"
IS_D_PRESENT="${IS_D_PRESENT:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "$AIRR_PATH" ]]; then
  echo "AIRR_PATH is required" >&2
  exit 1
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  echo "OUTPUT_PATH is required" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

cmd=(
  "$PYTHON_BIN" -m irrm_codec.calc_pgen_1mm
  --airr-path "$AIRR_PATH"
  --output-path "$OUTPUT_PATH"
  --chain "$CHAIN"
  --species "$SPECIES"
  --clone-id-col "$CLONE_ID_COL"
  --cdr3-col "$CDR3_COL"
  --threads "$THREADS"
  --chunk-size "$CHUNK_SIZE"
  --batch-size "$BATCH_SIZE"
  --pgen-col "$PGEN_COL"
  --log10-pgen-col "$LOG10_PGEN_COL"
  --is-d-present "$IS_D_PRESENT"
)

if [[ -n "$LOCUS" ]]; then
  cmd+=(--locus "$LOCUS")
fi

if [[ -n "$MODEL_PATH" ]]; then
  cmd+=(--model-path "$MODEL_PATH")
fi

if [[ -n "$MIRPY_PATH" ]]; then
  cmd+=(--mirpy-path "$MIRPY_PATH")
fi

echo "root_dir=$ROOT_DIR"
echo "airr_path=$AIRR_PATH"
echo "output_path=$OUTPUT_PATH"
echo "chain=$CHAIN"
echo "species=$SPECIES"
echo "locus=${LOCUS:-<none>}"
echo "threads=$THREADS chunk_size=$CHUNK_SIZE batch_size=$BATCH_SIZE"

"${cmd[@]}"
