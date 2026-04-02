# irrm-codec

Immune Receptor Rearrangement Model-based enCOder DECoder (IRRM-CODEC).

`irrm-codec` contains two neural models for working with TCR CDR3 amino-acid sequences and TCRemP embeddings:

- forward model: predicts a TCRemP embedding from CDR3 sequence input
- inverse model: reconstructs a CDR3 sequence from a TCRemP embedding

The repository is organized as a small training package with shell entrypoints and a notebook for quick inspection of results.

## Repository layout

- `irrm_codec/`: package with data loading, tokenization, datasets, models, losses, utilities and training entrypoints
- `scripts/`: shell wrappers for launching training
- `notebooks/`: example notebook for running training and analyzing saved metrics
- `artifacts/`: default output directory for checkpoints and run metadata

## Environment setup

Create the conda environment:

```bash
conda create -n irrm-codec python=3.11 -y
conda activate irrm-codec
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.4.1` and adds the PyTorch `cu121` wheel index to avoid pulling newer CUDA 13 builds that may require a newer NVIDIA driver.

Update the environment after dependency changes:

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name irrm-codec --display-name "Python (irrm-codec)"
```

Project dependencies, including notebook packages, are installed from [requirements.txt](/c:/Users/lizzka239/projects/irrm-codec/requirements.txt).

## Input data

Training expects two separate input files, following the same general idea as the `tcrempnet` workflow.

### 1. AIRR repertoire table

Accepted formats:

- `.tsv`
- `.airr`
- `.csv`
- `.parquet`

Required columns:

- `junction_aa`
- `v_call`
- `j_call`
- `locus`

Optional:

- `clone_id`

### 2. TCRemP embeddings parquet

Required:

- parquet file

Supported embedding layouts:

- one column with vector values, for example `tcremp_emb`
- many numeric embedding columns plus `clone_id`

If AIRR contains `clone_id`, the AIRR table and embeddings table are merged by `clone_id`. If AIRR does not contain `clone_id` but the two tables have the same number of rows, embeddings are matched to AIRR rows by row order.

## Training

### Forward model

```bash
bash scripts/train_forward.sh \
  --airr-path data/sample_airr.tsv \
  --embeddings-path data/sample_embeddings.parquet \
  --locus alpha \
  --output-dir artifacts/forward
```

### Inverse model

```bash
bash scripts/train_inverse.sh \
  --airr-path data/sample_airr.tsv \
  --embeddings-path data/sample_embeddings.parquet \
  --locus alpha \
  --output-dir artifacts/inverse
```

You can also run the modules directly:

```bash
python -m irrm_codec.train_forward \
  --airr-path data/sample_airr.tsv \
  --embeddings-path data/sample_embeddings.parquet \
  --locus alpha

python -m irrm_codec.train_inverse \
  --airr-path data/sample_airr.tsv \
  --embeddings-path data/sample_embeddings.parquet \
  --locus alpha
```

Useful optional flags:

- `--clone-id-col clone_id`
- `--embedding-column tcremp_emb`
- `--max-len 40`
- `--batch-size ...`
- `--epochs ...`
- `--seed 42`
- `--log-interval 10`
- `--no-progress`

## What the training scripts do

Both training scripts:

- load AIRR and embeddings from separate files
- align them by `clone_id` or by row order when `clone_id` is absent and row counts match
- filter by `locus`
- validate sequence and embedding inputs
- split data into train, validation and test subsets
- fit embedding normalization on the train split only
- save normalization parameters for later inference
- save both the best and the latest model checkpoints
- write per-epoch history and final test metrics

## Saved artifacts

Each run writes to its output directory, for example `artifacts/forward` or `artifacts/inverse`.

Saved files:

- `best.pt`: checkpoint with the best validation loss
- `last.pt`: checkpoint from the final epoch
- `mean.npy`: train-split embedding mean
- `std.npy`: train-split embedding standard deviation
- `history.json`: epoch-by-epoch training and validation metrics
- `test_metrics.json`: final metrics on the test split
- `data_stats.json`: dataset summary, merge statistics and artifact paths

## Notebook

The example notebook [notebooks/example_run_and_analysis.ipynb](/c:/Users/lizzka239/projects/irrm-codec/notebooks/example_run_and_analysis.ipynb) shows how to:

- inspect AIRR and embeddings inputs
- launch a training run from Python
- read saved metric files
- plot learning curves
- restore a checkpoint

## Notes

- The current inverse model predicts the full token sequence in parallel and uses a separate head for EOS-position prediction.
- The current pipeline expects one chain at a time and uses `locus` filtering to select it.
- Training outputs are intentionally ignored by git via [`.gitignore`](/c:/Users/lizzka239/projects/irrm-codec/.gitignore#L1).
