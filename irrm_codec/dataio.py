import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


AIRR_REQUIRED_COLUMNS = {"junction_aa", "v_call", "j_call", "locus"}


def read_airr_table(path, clone_id_col="clone_id", columns=None, validate=True):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path, columns=columns)
    elif suffix in {".tsv", ".airr"}:
        df = pd.read_csv(path, sep="\t", usecols=columns)
    elif suffix == ".csv":
        df = pd.read_csv(path, usecols=columns)
    else:
        raise ValueError(
            f"Unsupported AIRR file extension '{suffix}'. Use .tsv, .csv, .airr or .parquet."
        )

    if validate:
        missing_columns = AIRR_REQUIRED_COLUMNS.difference(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"AIRR table is missing required columns: {missing}")

        if clone_id_col in df.columns:
            if df[clone_id_col].isna().any():
                raise ValueError(f"AIRR table contains missing {clone_id_col} values.")
            if df[clone_id_col].duplicated().any():
                raise ValueError(f"AIRR table contains duplicate {clone_id_col} values.")

    return df.copy()
def open_embeddings_parquet(path):
    return pq.ParquetFile(Path(path))


def get_embedding_columns(parquet_file, clone_id_col="clone_id", embedding_column="tcremp_emb"):
    schema = parquet_file.schema_arrow
    column_names = schema.names

    if embedding_column in column_names:
        return [embedding_column], True

    numeric_columns = []
    for field in schema:
        if field.name == clone_id_col:
            continue
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            numeric_columns.append(field.name)

    if not numeric_columns:
        raise ValueError(
            f"Embeddings table must contain '{embedding_column}' or numeric embedding columns."
        )
    return numeric_columns, False


def iter_embedding_batches(
    path,
    batch_size=4096,
    clone_id_col="clone_id",
    embedding_column="tcremp_emb",
    include_clone_id=True,
):
    logger = logging.getLogger("irrm_codec")
    parquet_file = open_embeddings_parquet(path)
    embedding_columns, uses_nested_embedding = get_embedding_columns(
        parquet_file,
        clone_id_col=clone_id_col,
        embedding_column=embedding_column,
    )
    requested_columns = list(embedding_columns)
    has_clone_id = clone_id_col in parquet_file.schema_arrow.names
    if include_clone_id and has_clone_id and clone_id_col not in requested_columns:
        requested_columns.insert(0, clone_id_col)
    logger.info(
        "starting parquet batch iteration path=%s batch_size=%d include_clone_id=%s columns=%s",
        path,
        batch_size,
        include_clone_id,
        ",".join(requested_columns),
    )

    for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=requested_columns):
        batch_df = record_batch.to_pandas()
        if uses_nested_embedding:
            matrix = np.stack(batch_df[embedding_column].values).astype(np.float32)
            clone_ids = batch_df[clone_id_col].tolist() if include_clone_id and has_clone_id else None
        else:
            matrix = batch_df[embedding_columns].to_numpy(dtype=np.float32, copy=False)
            clone_ids = batch_df[clone_id_col].tolist() if include_clone_id and has_clone_id else None
        del batch_df
        yield clone_ids, matrix
    logger.info("finished parquet batch iteration path=%s", path)
