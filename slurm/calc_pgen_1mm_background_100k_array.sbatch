#!/usr/bin/env bash
#SBATCH --job-name=irrm-pgen-1mm-bg100k
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=/projects/immunestatus/vdjrearm/irrmcodec/logs/%x_%A_%a.out
#SBATCH --error=/projects/immunestatus/vdjrearm/irrmcodec/logs/%x_%A_%a.err

set -euo pipefail

cd ../

CHAINS=(TRB TRA TRG TRD IGH IGK IGL)
CHAIN="${CHAINS[$SLURM_ARRAY_TASK_ID]}"
CHAIN_LOWER="${CHAIN,,}"

case "$CHAIN" in
  TRA) LOCUS="alpha" ;;
  TRB) LOCUS="beta" ;;
  TRG) LOCUS="gamma" ;;
  TRD) LOCUS="delta" ;;
  IGH) LOCUS="heavy" ;;
  IGK) LOCUS="kappa" ;;
  IGL) LOCUS="lambda" ;;
  *)
    echo "Unsupported chain for locus mapping: $CHAIN" >&2
    exit 1
    ;;
esac

AIRR_DIR="/projects/immunestatus/vdjrearm/airr_format"
LOG_DIR="/projects/immunestatus/vdjrearm/irrmcodec/logs"
OUTPUT_ROOT="/projects/immunestatus/vdjrearm/pgen_1mm_background_100k"
MIRPY_PATH="${MIRPY_PATH:-}"

AIRR_PATH="${AIRR_DIR}/${CHAIN_LOWER}_background_100k.tsv"
OUTPUT_PATH="${OUTPUT_ROOT}/${CHAIN_LOWER}_background_100k_pgen.tsv"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-0}"
echo "chain=$CHAIN"
echo "species=human"
echo "airr_path=$AIRR_PATH"
echo "output_path=$OUTPUT_PATH"
echo "locus=$LOCUS"

"python" -m irrm_codec.calc_pgen_1mm \
  --airr-path "$AIRR_PATH" \
  --output-path "$OUTPUT_PATH" \
  --chain "$CHAIN" \
  --species human \
  --locus "$LOCUS" \
  --clone-id-col clone_id \
  --cdr3-col junction_aa \
  --threads "${SLURM_CPUS_PER_TASK:-32}" \
  --batch-size 2048 \
  --mirpy-path "$MIRPY_PATH"
