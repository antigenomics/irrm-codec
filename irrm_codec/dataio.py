from pathlib import Path

import numpy as np
import pandas as pd


AIRR_REQUIRED_COLUMNS = {"clone_id", "junction_aa", "v_call", "j_call", "locus"}


def read_airr_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".tsv", ".airr"}:
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported AIRR file extension '{suffix}'. Use .tsv, .csv, .airr or .parquet."
        )

    missing_columns = AIRR_REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"AIRR table is missing required columns: {missing}")
    if df["clone_id"].isna().any():
        raise ValueError("AIRR table contains missing clone_id values.")
    if df["clone_id"].duplicated().any():
        raise ValueError("AIRR table contains duplicate clone_id values.")

    return df.copy()


def extract_embedding_matrix(df, clone_id_col="clone_id", embedding_column="tcremp_emb"):
    if clone_id_col not in df.columns:
        raise ValueError(f"Embeddings table must contain '{clone_id_col}'.")
    if df[clone_id_col].isna().any():
        raise ValueError("Embeddings table contains missing clone_id values.")
    if df[clone_id_col].duplicated().any():
        raise ValueError("Embeddings table contains duplicate clone_id values.")

    if embedding_column in df.columns:
        matrix = np.stack(df[embedding_column].values).astype(np.float32)
        embedding_df = df[[clone_id_col]].copy()
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
        embedding_df = df[[clone_id_col]].copy()

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings matrix contains NaN or infinite values.")

    return embedding_df, matrix


def load_airr_with_embeddings(
    airr_path,
    embeddings_path,
    locus=None,
    clone_id_col="clone_id",
    embedding_column="tcremp_emb",
):
    airr_df = read_airr_table(airr_path)
    if locus is not None:
        airr_df = airr_df[airr_df["locus"] == locus].reset_index(drop=True)
    if len(airr_df) == 0:
        raise ValueError("AIRR table is empty after locus filtering.")

    embeddings_raw = pd.read_parquet(embeddings_path)
    embeddings_df, emb_matrix = extract_embedding_matrix(
        embeddings_raw,
        clone_id_col=clone_id_col,
        embedding_column=embedding_column,
    )
    embeddings_df["_embedding_index"] = np.arange(len(embeddings_df))

    merged = airr_df.merge(
        embeddings_df,
        left_on=clone_id_col,
        right_on=clone_id_col,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) == 0:
        raise ValueError("No rows matched between AIRR and embeddings tables by clone_id.")

    emb = emb_matrix[merged["_embedding_index"].to_numpy()]
    merged = merged.drop(columns=["_embedding_index"]).reset_index(drop=True)

    stats = {
        "airr_rows": int(len(airr_df)),
        "embeddings_rows": int(len(embeddings_df)),
        "merged_rows": int(len(merged)),
        "airr_unmatched_rows": int(len(airr_df) - len(merged)),
        "embeddings_unmatched_rows": int(len(embeddings_df) - len(merged)),
        "clone_id_column": clone_id_col,
        "embedding_column": embedding_column if embedding_column in embeddings_raw.columns else None,
    }
    return merged, emb, stats
