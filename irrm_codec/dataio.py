from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from irrm_codec.utils import apply_standardizer


AIRR_REQUIRED_COLUMNS = {"junction_aa", "v_call", "j_call", "locus"}


def _make_split_labels(num_items, train_fraction=0.8, val_fraction=0.1, seed=42):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_items)
    labels = np.full(num_items, 2, dtype=np.int8)
    train_end = int(num_items * train_fraction)
    val_end = train_end + int(num_items * val_fraction)
    labels[perm[:train_end]] = 0
    labels[perm[train_end:val_end]] = 1
    return labels


def _iter_airr_chunks(path, chunk_size):
    path = Path(path)
    suffix = path.suffix.lower()
    columns = sorted(AIRR_REQUIRED_COLUMNS)
    if suffix == ".parquet":
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
            yield batch.to_pandas()
    elif suffix in {".tsv", ".airr"}:
        yield from pd.read_csv(path, sep="\t", usecols=columns, chunksize=chunk_size)
    elif suffix == ".csv":
        yield from pd.read_csv(path, usecols=columns, chunksize=chunk_size)
    else:
        raise ValueError(
            f"Unsupported AIRR file extension '{suffix}'. Use .tsv, .csv, .airr or .parquet."
        )


def _iter_embedding_chunks(path, chunk_size, embedding_column="tcremp_emb"):
    parquet_file = pq.ParquetFile(path)
    schema_names = set(parquet_file.schema.names)
    if embedding_column in schema_names:
        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=[embedding_column]):
            chunk_df = batch.to_pandas()
            matrix = np.stack(chunk_df[embedding_column].values).astype(np.float32)
            if matrix.ndim != 2:
                raise ValueError(f"Expected 2D embeddings matrix, got shape {matrix.shape}.")
            if not np.isfinite(matrix).all():
                raise ValueError("Embeddings matrix contains NaN or infinite values.")
            yield matrix
        return

    numeric_columns = []
    for name in parquet_file.schema.names:
        field = parquet_file.schema_arrow.field(name)
        if str(field.type) in {"float", "double", "int8", "int16", "int32", "int64"}:
            numeric_columns.append(name)
    if not numeric_columns:
        raise ValueError(
            f"Embeddings table must contain '{embedding_column}' or numeric embedding columns."
        )

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=numeric_columns):
        chunk_df = batch.to_pandas()
        matrix = chunk_df[numeric_columns].to_numpy(dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D embeddings matrix, got shape {matrix.shape}.")
        if not np.isfinite(matrix).all():
            raise ValueError("Embeddings matrix contains NaN or infinite values.")
        yield matrix


def _count_filtered_airr_rows(path, locus=None, chunk_size=100_000):
    total_rows = 0
    filtered_rows = 0
    for chunk in _iter_airr_chunks(path, chunk_size):
        missing_columns = AIRR_REQUIRED_COLUMNS.difference(chunk.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"AIRR table is missing required columns: {missing}")
        total_rows += len(chunk)
        if locus is None:
            filtered_rows += len(chunk)
        else:
            filtered_rows += int(chunk["locus"].eq(locus).sum())
    if filtered_rows == 0:
        raise ValueError("AIRR table is empty after locus filtering.")
    return total_rows, filtered_rows


def _update_sequence_stats(stats, sequences, max_len):
    for raw_seq in sequences.tolist():
        seq = "" if raw_seq is None else str(raw_seq).strip().upper()
        if not seq:
            stats["empty_sequences"] += 1
            continue
        seq_len = len(seq)
        stats["sequence_length_sum"] += seq_len
        stats["min_length"] = seq_len if stats["min_length"] is None else min(stats["min_length"], seq_len)
        stats["max_length"] = seq_len if stats["max_length"] is None else max(stats["max_length"], seq_len)
        stats["unk_sequences"] += int(any(char not in "ACDEFGHIKLMNPQRSTVWY" for char in seq))
        stats["truncated_sequences"] += int(seq_len > max_len)


def _new_shard_writer(split_name, temp_dir, shard_size):
    return {
        "split_name": split_name,
        "temp_dir": Path(temp_dir),
        "shard_size": shard_size,
        "pending_sequences": [],
        "pending_embeddings": [],
        "pending_rows": 0,
        "shard_index": 0,
        "shards": [],
        "num_rows": 0,
    }


def _flush_shard(writer, rows_to_write):
    seq_chunks = []
    emb_chunks = []
    remaining = rows_to_write

    while remaining > 0:
        seq_chunk = writer["pending_sequences"][0]
        emb_chunk = writer["pending_embeddings"][0]
        take = min(len(seq_chunk), remaining)
        seq_chunks.append(seq_chunk[:take])
        emb_chunks.append(emb_chunk[:take])
        if take == len(seq_chunk):
            writer["pending_sequences"].pop(0)
            writer["pending_embeddings"].pop(0)
        else:
            writer["pending_sequences"][0] = seq_chunk[take:]
            writer["pending_embeddings"][0] = emb_chunk[take:]
        writer["pending_rows"] -= take
        remaining -= take

    sequences = np.concatenate(seq_chunks)
    embeddings = np.concatenate(emb_chunks, axis=0).astype(np.float32, copy=False)
    shard_prefix = writer["temp_dir"] / f"{writer['split_name']}_shard_{writer['shard_index']:05d}"
    seq_path = Path(f"{shard_prefix}.seq.npy")
    emb_path = Path(f"{shard_prefix}.emb.npy")
    np.save(seq_path, sequences, allow_pickle=False)
    np.save(emb_path, embeddings, allow_pickle=False)
    writer["shards"].append(
        {
            "seq_path": str(seq_path),
            "emb_path": str(emb_path),
            "num_rows": int(len(sequences)),
        }
    )
    writer["shard_index"] += 1


def _writer_add(writer, sequences, embeddings):
    if len(sequences) == 0:
        return
    writer["pending_sequences"].append(np.asarray(sequences))
    writer["pending_embeddings"].append(np.asarray(embeddings, dtype=np.float32))
    writer["pending_rows"] += len(sequences)
    writer["num_rows"] += len(sequences)
    while writer["pending_rows"] >= writer["shard_size"]:
        _flush_shard(writer, writer["shard_size"])


def _writer_finalize(writer):
    if writer["pending_rows"] > 0:
        _flush_shard(writer, writer["pending_rows"])
    return writer["shards"]


def _first_pass(
    airr_path,
    embeddings_path,
    locus,
    embedding_column,
    train_fraction,
    val_fraction,
    seed,
    chunk_size,
    max_len,
):
    raw_airr_rows, filtered_rows = _count_filtered_airr_rows(airr_path, locus=locus, chunk_size=chunk_size)
    split_labels = _make_split_labels(
        filtered_rows,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )

    train_sum = None
    train_sum_sq = None
    train_count = 0
    embedding_dim = None
    filtered_cursor = 0
    embedding_rows_seen = 0
    stats = {
        "num_samples": int(filtered_rows),
        "embedding_dim": None,
        "num_unique_clone_ids": int(filtered_rows),
        "min_length": None,
        "max_length": None,
        "sequence_length_sum": 0,
        "truncated_sequences": 0,
        "unk_sequences": 0,
        "empty_sequences": 0,
        "max_len": max_len,
    }
    split_counts = {"train_size": 0, "val_size": 0, "test_size": 0}

    airr_iter = _iter_airr_chunks(airr_path, chunk_size)
    emb_iter = _iter_embedding_chunks(embeddings_path, chunk_size, embedding_column=embedding_column)
    for airr_chunk, emb_chunk in zip_longest(airr_iter, emb_iter, fillvalue=None):
        if airr_chunk is None or emb_chunk is None:
            raise ValueError("AIRR and embeddings files have different numbers of rows.")
        if len(airr_chunk) != len(emb_chunk):
            raise ValueError("AIRR and embeddings chunks are misaligned by row count.")

        embedding_rows_seen += len(emb_chunk)
        if embedding_dim is None:
            embedding_dim = int(emb_chunk.shape[1])
            stats["embedding_dim"] = embedding_dim
            train_sum = np.zeros(embedding_dim, dtype=np.float64)
            train_sum_sq = np.zeros(embedding_dim, dtype=np.float64)
        elif emb_chunk.shape[1] != embedding_dim:
            raise ValueError(
                f"Inconsistent embedding dimension: expected {embedding_dim}, got {emb_chunk.shape[1]}."
            )

        keep_mask = np.ones(len(airr_chunk), dtype=bool) if locus is None else airr_chunk["locus"].to_numpy() == locus
        if not keep_mask.any():
            continue

        filtered_airr = airr_chunk.loc[keep_mask, sorted(AIRR_REQUIRED_COLUMNS)].reset_index(drop=True)
        filtered_emb = emb_chunk[keep_mask]
        sequences = filtered_airr["junction_aa"].astype(str).to_numpy()

        _update_sequence_stats(stats, sequences, max_len=max_len)
        labels = split_labels[filtered_cursor : filtered_cursor + len(filtered_airr)]
        filtered_cursor += len(filtered_airr)

        train_mask = labels == 0
        val_mask = labels == 1
        test_mask = labels == 2

        if train_mask.any():
            train_emb = filtered_emb[train_mask].astype(np.float32, copy=False)
            train_sum += train_emb.sum(axis=0, dtype=np.float64)
            train_sum_sq += np.square(train_emb, dtype=np.float32).sum(axis=0, dtype=np.float64)
            count = int(train_mask.sum())
            train_count += count
            split_counts["train_size"] += count
        if val_mask.any():
            split_counts["val_size"] += int(val_mask.sum())
        if test_mask.any():
            split_counts["test_size"] += int(test_mask.sum())

    if filtered_cursor != filtered_rows:
        raise ValueError("Filtered AIRR row count changed during first pass.")
    if embedding_rows_seen != raw_airr_rows:
        raise ValueError("AIRR and embeddings files have different total row counts.")
    if stats["empty_sequences"]:
        raise ValueError(f"Found {stats['empty_sequences']} empty or missing sequences.")
    if train_count == 0:
        raise ValueError("Train split is empty after filtering.")

    mean = train_sum / train_count
    variance = np.maximum(train_sum_sq / train_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-8, 1.0, std)

    return {
        "raw_airr_rows": int(raw_airr_rows),
        "filtered_rows": int(filtered_rows),
        "split_labels": split_labels,
        "stats": stats,
        "split_counts": split_counts,
        "embedding_rows_seen": int(embedding_rows_seen),
        "embedding_dim": int(embedding_dim),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def _second_pass_write_shards(
    airr_path,
    embeddings_path,
    locus,
    embedding_column,
    chunk_size,
    split_labels,
    mean,
    std,
    shard_size,
    output_temp_dir,
):
    train_writer = _new_shard_writer("train", output_temp_dir, shard_size)
    val_writer = _new_shard_writer("val", output_temp_dir, shard_size)
    test_writer = _new_shard_writer("test", output_temp_dir, shard_size)
    filtered_cursor = 0
    embedding_rows_seen = 0

    airr_iter = _iter_airr_chunks(airr_path, chunk_size)
    emb_iter = _iter_embedding_chunks(embeddings_path, chunk_size, embedding_column=embedding_column)
    for airr_chunk, emb_chunk in zip_longest(airr_iter, emb_iter, fillvalue=None):
        if airr_chunk is None or emb_chunk is None:
            raise ValueError("AIRR and embeddings files have different numbers of rows.")
        if len(airr_chunk) != len(emb_chunk):
            raise ValueError("AIRR and embeddings chunks are misaligned by row count.")

        embedding_rows_seen += len(emb_chunk)
        keep_mask = np.ones(len(airr_chunk), dtype=bool) if locus is None else airr_chunk["locus"].to_numpy() == locus
        if not keep_mask.any():
            continue

        filtered_airr = airr_chunk.loc[keep_mask, sorted(AIRR_REQUIRED_COLUMNS)].reset_index(drop=True)
        filtered_emb = apply_standardizer(emb_chunk[keep_mask], mean, std).astype(np.float32, copy=False)
        sequences = filtered_airr["junction_aa"].astype(str).to_numpy()
        labels = split_labels[filtered_cursor : filtered_cursor + len(filtered_airr)]
        filtered_cursor += len(filtered_airr)

        train_mask = labels == 0
        val_mask = labels == 1
        test_mask = labels == 2

        if train_mask.any():
            _writer_add(train_writer, sequences[train_mask], filtered_emb[train_mask])
        if val_mask.any():
            _writer_add(val_writer, sequences[val_mask], filtered_emb[val_mask])
        if test_mask.any():
            _writer_add(test_writer, sequences[test_mask], filtered_emb[test_mask])

    return {
        "train_shards": _writer_finalize(train_writer),
        "val_shards": _writer_finalize(val_writer),
        "test_shards": _writer_finalize(test_writer),
        "embedding_rows_seen": int(embedding_rows_seen),
        "filtered_cursor": int(filtered_cursor),
    }


def prepare_row_aligned_sharded_data(
    airr_path,
    embeddings_path,
    output_temp_dir,
    locus=None,
    embedding_column="tcremp_emb",
    train_fraction=0.8,
    val_fraction=0.1,
    seed=42,
    shard_size=100_000,
    chunk_size=100_000,
    max_len=40,
):
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    first_pass = _first_pass(
        airr_path=airr_path,
        embeddings_path=embeddings_path,
        locus=locus,
        embedding_column=embedding_column,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
        chunk_size=chunk_size,
        max_len=max_len,
    )
    second_pass = _second_pass_write_shards(
        airr_path=airr_path,
        embeddings_path=embeddings_path,
        locus=locus,
        embedding_column=embedding_column,
        chunk_size=chunk_size,
        split_labels=first_pass["split_labels"],
        mean=first_pass["mean"],
        std=first_pass["std"],
        shard_size=shard_size,
        output_temp_dir=output_temp_dir,
    )

    if second_pass["filtered_cursor"] != first_pass["filtered_rows"]:
        raise ValueError("Filtered AIRR row count changed between passes.")
    if second_pass["embedding_rows_seen"] != first_pass["raw_airr_rows"]:
        raise ValueError("AIRR and embeddings files have different total row counts.")

    stats = first_pass["stats"]
    split_counts = first_pass["split_counts"]
    data_stats = {
        "num_samples": stats["num_samples"],
        "embedding_dim": stats["embedding_dim"],
        "num_unique_clone_ids": stats["num_unique_clone_ids"],
        "min_length": int(stats["min_length"]),
        "max_length": int(stats["max_length"]),
        "mean_length": float(stats["sequence_length_sum"] / stats["num_samples"]),
        "truncated_fraction": stats["truncated_sequences"] / stats["num_samples"],
        "unk_sequence_fraction": stats["unk_sequences"] / stats["num_samples"],
        "max_len": max_len,
    }
    split_stats = {
        **{key: int(value) for key, value in split_counts.items()},
        "train_num_shards": int(len(second_pass["train_shards"])),
        "val_num_shards": int(len(second_pass["val_shards"])),
        "test_num_shards": int(len(second_pass["test_shards"])),
    }
    merge_stats = {
        "airr_rows": int(first_pass["raw_airr_rows"]),
        "embeddings_rows": int(first_pass["embedding_rows_seen"]),
        "merged_rows": int(first_pass["filtered_rows"]),
        "airr_unmatched_rows": int(first_pass["raw_airr_rows"] - first_pass["filtered_rows"]),
        "embeddings_unmatched_rows": int(first_pass["embedding_rows_seen"] - first_pass["filtered_rows"]),
        "clone_id_column": None,
        "embedding_column": embedding_column,
        "alignment_mode": "row_order",
    }
    return {
        "data_stats": data_stats,
        "split_stats": split_stats,
        "merge_stats": merge_stats,
        "mean": first_pass["mean"],
        "std": first_pass["std"],
        "embedding_dim": first_pass["embedding_dim"],
        "train_shard_paths": second_pass["train_shards"],
        "val_shard_paths": second_pass["val_shards"],
        "test_shard_paths": second_pass["test_shards"],
    }
