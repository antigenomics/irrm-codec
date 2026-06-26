#!/usr/bin/env bash
#SBATCH --job-name=irrm-inverse-sweep
#SBATCH --partition=short
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/immunestatus/vdjrearm/irrmcodec/logs/%x_%A_%a.out
#SBATCH --error=/projects/immunestatus/vdjrearm/irrmcodec/logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

CHAIN="${CHAIN:?CHAIN is required}"
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

PYTHON_BIN="${PYTHON_BIN:-python}"
AIRR_DIR="${AIRR_DIR:-/projects/immunestatus/vdjrearm/airr_format}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-/projects/immunestatus/vdjrearm/tcremp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/projects/immunestatus/vdjrearm/irrmcodec/wandb_sweeps/inverse}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_SLUG="${RUN_SLUG:?RUN_SLUG is required}"
WANDB_PROJECT="${WANDB_PROJECT:-irrm-codec-inverse-${CHAIN_LOWER}}"

AIRR_PATH="${AIRR_DIR}/${CHAIN_LOWER}_background_100k.tsv"
EMBEDDINGS_PATH="${EMBEDDINGS_DIR}/${CHAIN_LOWER}_background_100k_tcremp.parquet"
OUTPUT_DIR="${OUTPUT_ROOT}/${CHAIN_LOWER}/${RUN_SLUG}"

mkdir -p "$OUTPUT_DIR"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "chain=$CHAIN"
echo "locus=$LOCUS"
echo "run_slug=$RUN_SLUG"
echo "wandb_project=$WANDB_PROJECT"
echo "output_dir=$OUTPUT_DIR"

"$PYTHON_BIN" -m irrm_codec.train_inverse \
  --airr-path "$AIRR_PATH" \
  --embeddings-path "$EMBEDDINGS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --locus "$LOCUS" \
  --clone-id-col "${CLONE_ID_COL:-clone_id}" \
  --embedding-column "${EMBEDDING_COLUMN:-tcremp_emb}" \
  --max-len "${MAX_LEN:-40}" \
  --hidden-dim "${HIDDEN_DIM:-512}" \
  --dropout "${DROPOUT:-0.2}" \
  --num-layers "${NUM_LAYERS:-3}" \
  --nhead "${NHEAD:-8}" \
  --ff-mult "${FF_MULT:-4}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --epochs "${EPOCHS:-40}" \
  --lr "${LR:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --train-fraction "${TRAIN_FRACTION:-0.8}" \
  --val-fraction "${VAL_FRACTION:-0.1}" \
  --seed "${SEED:-42}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-run-name "$RUN_SLUG" \
  --wandb-dir "$OUTPUT_DIR" \
  --wandb-mode "$WANDB_MODE" \
  --log-interval "${LOG_INTERVAL:-10}"
