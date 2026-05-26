#!/usr/bin/env bash
#SBATCH --job-name=irrm-pgen-trb
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/immunestatus/vdjrearm/irrmcodec/logs/%x_%j.out
#SBATCH --error=/projects/immunestatus/vdjrearm/irrmcodec/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

AIRR_PATH="/projects/immunestatus/vdjrearm/airr_format/trb_background_100k.tsv"
OUTPUT_PATH="/projects/immunestatus/vdjrearm/pgen_1mm_background_100k/trb_background_100k_pgen.tsv"
LOG_DIR="/projects/immunestatus/vdjrearm/irrmcodec/logs"
OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "root_dir=$ROOT_DIR"
echo "airr_path=$AIRR_PATH"
echo "output_path=$OUTPUT_PATH"

python -m irrm_codec.calc_pgen_1mm \
  --airr-path "$AIRR_PATH" \
  --output-path "$OUTPUT_PATH" \
  --chain TRB \
  --species human \
  --locus beta \
  --threads "${SLURM_CPUS_PER_TASK:-8}"
