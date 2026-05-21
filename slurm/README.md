`slurm/train_forward.sbatch` launches the same forward training you ran from the notebook, but through `sbatch`.

Default paths in the job already match your current TRB run:

- AIRR: `/projects/immunestatus/vdjdb/airr_format/trb_background.tsv`
- embeddings: `/projects/immunestatus/vdjdb/tcremp/trb_background_embeddings.parquet`
- output: `artifacts/forward_demo_trb`
- locus: `beta`

Submit as-is:

```bash
sbatch slurm/train_forward.sbatch
```

Override paths or hyperparameters without editing the file:

```bash
sbatch \
  --export=ALL,OUTPUT_DIR=/home/evlasova/irrm-codec/artifacts/forward_1m_trb,EPOCHS=10,BATCH_SIZE=512,NUM_WORKERS=16,PYTHON_BIN=/home/evlasova/.conda/envs/irrm-codec/bin/python \
  slurm/train_forward.sbatch
```

If your cluster needs a different queue or resources, edit the `#SBATCH` lines at the top of the script:

- `--partition`
- `--gres`
- `--cpus-per-task`
- `--mem`
- `--time`

Logs go to `slurm/logs/`.

`slurm/calc_pgen_1mm.sbatch` runs 1-mismatch pgen calculation on an AIRR table using
`mirpy`'s OLGA wrapper.

Example:

```bash
sbatch \
  --export=ALL,AIRR_PATH=/projects/immunestatus/vdjrearm/airr_format/trb_background_100k.tsv,OUTPUT_PATH=/projects/immunestatus/vdjrearm/pgen/trb_background_100k_pgen.tsv,CHAIN=TRB,LOCUS=beta,THREADS=8,BATCH_SIZE=2048,PYTHON_BIN=/home/evlasova/.conda/envs/irrm-codec/bin/python \
  slurm/calc_pgen_1mm.sbatch
```

Important environment variables:

- `AIRR_PATH`: input AIRR table
- `OUTPUT_PATH`: output file (`.tsv`, `.airr`, `.csv`, or `.parquet`)
- `CHAIN`: OLGA chain name, default `TRB`
- `SPECIES`: OLGA species name, default `human`
- `LOCUS`: optional AIRR locus filter
- `MODEL_PATH`: optional explicit OLGA model directory
- `MIRPY_PATH`: optional local mirpy checkout, default `../mirpy`
- `THREADS`: number of worker threads
- `BATCH_SIZE`: sequences per inner batch

`slurm/calc_pgen_1mm_background_100k_array.sbatch` computes `1mm pgen` for all seven chains
in the default background-100k layout via a Slurm array.

Run from the repository root:

```bash
sbatch slurm/calc_pgen_1mm_background_100k_array.sbatch
```

`slurm/train_pgen_background_100k_array.sbatch` trains one sequence-to-`log10_pgen_1mm` model
per chain, assuming the pgen tables already exist.

Run from the repository root:

```bash
sbatch slurm/train_pgen_background_100k_array.sbatch
```
