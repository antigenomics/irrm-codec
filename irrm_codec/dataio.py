import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


AIRR_REQUIRED_COLUMNS = {"junction_aa", "v_call", "j_call", "locus"}

LOCUS_ALIASES = {
    "tra": "alpha",
    "trb": "beta",
    "trg": "gamma",
    "trd": "delta",
    "igh": "heavy",
    "igk": "kappa",
    "igl": "lambda",
}


def normalize_locus_name(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return LOCUS_ALIASES.get(normalized, normalized)


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


def extract_embedding_matrix(df, clone_id_col="clone_id", embedding_column="tcremp_emb"):
    if embedding_column in df.columns:
        matrix = np.stack(df[embedding_column].values).astype(np.float32)
    else:
        numeric_columns = [
            column
            for column in df.columns
            if column != clone_id_col and pd.api.types.is_numeric_dtype(df[column])
        ]
        if not numeric_columns:
            raise ValueError(
                f"Embeddings table must contain '{embedding_column}' or numeric embedding columns."
            )
        matrix = df[numeric_columns].to_numpy(dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings matrix contains NaN or infinite values.")
    return matrix


def load_airr_with_embeddings(
    airr_path,
    embeddings_path,
    locus=None,
    clone_id_col="clone_id",
    embedding_column="tcremp_emb",
):
    airr_df = read_airr_table(airr_path, clone_id_col=clone_id_col)
    airr_rows_before_locus = len(airr_df)
    if locus is not None:
        locus = normalize_locus_name(locus)
        locus_series = airr_df["locus"].astype(str).str.strip().str.lower().map(normalize_locus_name)
        airr_df = airr_df[locus_series == locus].reset_index(drop=True)
    if len(airr_df) == 0:
        raise ValueError("AIRR table is empty after locus filtering.")

    embeddings_raw = pd.read_parquet(embeddings_path)
    can_align_by_row_order = (
        len(airr_df) == len(embeddings_raw) and clone_id_col not in embeddings_raw.columns
    )
    if can_align_by_row_order:
        merged = airr_df.copy().reset_index(drop=True)
        if clone_id_col not in merged.columns:
            merged[clone_id_col] = np.arange(len(merged)).astype(str)
        emb = extract_embedding_matrix(
            embeddings_raw,
            clone_id_col=clone_id_col,
            embedding_column=embedding_column,
        )
        use_row_alignment = True
    else:
        if clone_id_col not in embeddings_raw.columns:
            raise ValueError(
                "Embeddings table does not contain "
                f"'{clone_id_col}', so merge by id is impossible. "
                f"Row-order alignment is only allowed when row counts match exactly: "
                f"AIRR before locus filter={airr_rows_before_locus}, "
                f"AIRR after locus filter={len(airr_df)}, "
                f"embeddings={len(embeddings_raw)}, "
                f"locus={locus!r}."
            )
        if embeddings_raw[clone_id_col].isna().any():
            raise ValueError("Embeddings table contains missing clone_id values.")
        if embeddings_raw[clone_id_col].duplicated().any():
            raise ValueError("Embeddings table contains duplicate clone_id values.")

        emb_matrix = extract_embedding_matrix(
            embeddings_raw,
            clone_id_col=clone_id_col,
            embedding_column=embedding_column,
        )
        embeddings_df = embeddings_raw[[clone_id_col]].copy()
        embeddings_df["_embedding_index"] = np.arange(len(embeddings_df))

        merged = airr_df.merge(
            embeddings_df,
            left_on=clone_id_col,
            right_on=clone_id_col,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) == 0:
            raise ValueError(f"No rows matched between AIRR and embeddings tables by {clone_id_col}.")

        emb = emb_matrix[merged["_embedding_index"].to_numpy()]
        merged = merged.drop(columns=["_embedding_index"]).reset_index(drop=True)
        use_row_alignment = False

    stats = {
        "airr_rows": int(len(airr_df)),
        "airr_rows_before_locus": int(airr_rows_before_locus),
        "embeddings_rows": int(len(embeddings_raw)),
        "merged_rows": int(len(merged)),
        "airr_unmatched_rows": int(len(airr_df) - len(merged)),
        "embeddings_unmatched_rows": int(len(embeddings_raw) - len(merged)),
        "clone_id_column": clone_id_col,
        "alignment_mode": "row_order" if use_row_alignment else "clone_id",
    }
    return merged, emb, stats
