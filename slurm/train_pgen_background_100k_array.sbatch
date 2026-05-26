#!/usr/bin/env bash
#SBATCH --job-name=irrm-train-pgen-bg100k
#SBATCH --partition=short
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --output=/projects/immunestatus/vdjrearm/irrmcodec/logs/%x_%A_%a.out
#SBATCH --error=/projects/immunestatus/vdjrearm/irrmcodec/logs/%x_%A_%a.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

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

AIRR_DIR="/projects/immunestatus/vdjrearm/pgen_1mm_background_100k"
LOG_DIR="/projects/immunestatus/vdjrearm/irrmcodec/logs"
OUTPUT_ROOT="/projects/immunestatus/vdjrearm/irrmcodec/pgen_background_100k"

AIRR_PATH="${AIRR_DIR}/${CHAIN_LOWER}_background_100k_pgen.tsv"
OUTPUT_DIR="${OUTPUT_ROOT}/${CHAIN_LOWER}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID:-0}"
echo "root_dir=$ROOT_DIR"
echo "chain=$CHAIN"
echo "airr_path=$AIRR_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "locus=$LOCUS"

"python" -m irrm_codec.train_pgen \
  --airr-path "$AIRR_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --locus "$LOCUS" \
  --target-col log10_pgen_1mm \
  --clone-id-col clone_id \
  --max-len 40 \
  --batch-size 256 \
  --epochs 40 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --dropout 0.2 \
  --encoder-type plain_conv \
  --hidden-dim 192 \
  --mlp-dim 512 \
  --mlp-hidden-dim 1024 \
  --dilations 1,2,4,8 \
  --early-stopping-patience 5 \
  --scheduler plateau \
  --scheduler-patience 2 \
  --scheduler-factor 0.5 \
  --train-fraction 0.8 \
  --val-fraction 0.1 \
  --seed 42 \
  --num-workers 8 \
  --log-interval 10 \
  --no-progress
