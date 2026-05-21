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
For pgen workflows, it also installs `mirpy-lib`, the immunosequencing package published from `antigenomics/mirpy`.

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

## 1mm pgen calculation

Use the dedicated module to compute 1-mismatch pgen values through `mirpy`'s
`mir.basic.pgen.OlgaModel.compute_pgen_junction_aa_1mm`.

Example:

```bash
python -m irrm_codec.calc_pgen_1mm \
  --airr-path data/sample_airr.tsv \
  --output-path artifacts/pgen/sample_airr_pgen.tsv \
  --chain TRB \
  --species human \
  --locus beta \
  --threads 8 \
  --batch-size 1024
```

Notes:

- The input AIRR file is split into contiguous chunks across `--threads`.
- Each worker thread creates its OLGA model lazily and reuses it for all batches in that thread.
- The output table keeps the original AIRR columns and appends `pgen_1mm` and `log10_pgen_1mm`.
- The code is compatible with the current `mirpy-lib` package and falls back to a local checkout at `../mirpy` when needed.

## Sequence to log10(pgen) regression

Use the dedicated training module to fit a scalar regressor from CDR3 sequence to `log10_pgen_1mm`.

Example:

```bash
python -m irrm_codec.train_pgen \
  --airr-path artifacts/pgen/sample_airr_pgen.tsv \
  --output-dir artifacts/pgen_model/trb \
  --target-col log10_pgen_1mm \
  --locus beta \
  --batch-size 256 \
  --epochs 40
```

Notes:

- The model reuses the sequence encoder structure from the forward model and predicts one scalar per clonotype.
- The training target is taken directly from the AIRR table, so pgen preprocessing must be run first.
- Metrics include Huber loss, RMSE, and MAE.

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

- The inverse model predicts the full fixed-length token sequence of length 40 in parallel.
- The current pipeline expects one chain at a time and uses `locus` filtering to select it.
- By default, CDR3 sequences are converted to fixed length 40 before tokenization by inserting `-` gaps after residue 4 and before the last 3 residues.
- Training outputs are intentionally ignored by git via [`.gitignore`](/c:/Users/lizzka239/projects/irrm-codec/.gitignore#L1).
