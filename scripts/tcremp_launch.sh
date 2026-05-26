#!/bin/bash
#SBATCH --job-name=tcremp
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-4
#SBATCH --output=/projects/immunestatus/vdjrearm/tcremp/logs/%x_%A_%a.out
#SBATCH --error=/projects/immunestatus/vdjrearm/tcremp/logs/%x_%A_%a.err

set -euo pipefail

CHAINS=(TRG TRD IGH IGK IGL)
CHAIN="${CHAINS[$SLURM_ARRAY_TASK_ID]}"
CHAIN_LOWER="${CHAIN,,}"

INPUT_DIR="/projects/immunestatus/vdjrearm/airr_format"
INPUT="${INPUT_DIR}/${CHAIN_LOWER}_background_100k.tsv"
OUTDIR="/projects/immunestatus/vdjrearm/tcremp/"
MIN_CDR3_LEN=7
MAX_CDR3_LEN=40

mkdir -p "$OUTDIR" /projects/immunestatus/vdjrearm/tcremp/logs


tcremp-run \
  -i "$INPUT" \
  -o "$OUTDIR" \
  -c "$CHAIN" \
  -s HomoSapiens \
  -np "$SLURM_CPUS_PER_TASK" \
  --lower-len-cdr3 "$MIN_CDR3_LEN" \
  --higher-len-cdr3 "$MAX_CDR3_LEN" \
  --skip-clustering
