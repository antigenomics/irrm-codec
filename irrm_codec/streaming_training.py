import logging

import numpy as np
from torch.utils.data import DataLoader

from irrm_codec.dataio import inspect_embeddings_file, iter_embedding_batches, read_airr_table
from irrm_codec.datasets import StreamingEmbeddingDataset
from irrm_codec.utils import split_indices


def load_airr_records(args):
    logger = logging.getLogger("irrm_codec")
    logger.info("reading AIRR table from %s", args.airr_path)
    airr_df = read_airr_table(args.airr_path, clone_id_col=args.clone_id_col)
    if args.locus is not None:
        logger.info("filtering AIRR table by locus=%s", args.locus)
        airr_df = airr_df[airr_df["locus"] == args.locus].reset_index(drop=True)
    if len(airr_df) == 0:
        raise ValueError("AIRR table is empty after locus filtering.")
    logger.info("finished AIRR read rows=%d", len(airr_df))
    return airr_df


def build_embedding_iterator(args, alignment_mode):
    def iterator():
        row_offset = 0
        logger = logging.getLogger("irrm_codec")
        logger.info(
            "starting embeddings iterator path=%s alignment_mode=%s reader_batch_size=%d",
            args.embeddings_path,
            alignment_mode,
            args.reader_batch_size,
        )
        for clone_ids, emb_batch in iter_embedding_batches(
            args.embeddings_path,
            batch_size=args.reader_batch_size,
            clone_id_col=args.clone_id_col,
            embedding_column=args.embedding_column,
            include_clone_id=(alignment_mode == "clone_id"),
        ):
            if alignment_mode == "row_order":
                keys = list(range(row_offset, row_offset + len(emb_batch)))
                row_offset += len(emb_batch)
            else:
                keys = clone_ids
            yield keys, emb_batch
        logger.info("finished embeddings iterator alignment_mode=%s", alignment_mode)

    return iterator


def prepare_streaming_splits(args, logger=None):
    logger = logger or logging.getLogger("irrm_codec")
    logger.info("prepare_streaming_splits: start")
    airr_df = load_airr_records(args)
    logger.info("loaded AIRR table rows=%d after locus filtering", len(airr_df))
    logger.info("inspecting embeddings file metadata path=%s", args.embeddings_path)
    embedding_info = inspect_embeddings_file(
        args.embeddings_path,
        clone_id_col=args.clone_id_col,
        embedding_column=args.embedding_column,
    )
    logger.info(
        "inspected embeddings file rows=%d embedding_dim=%d has_clone_id=%s",
        embedding_info["num_rows"],
        embedding_info["embedding_dim"],
        embedding_info["has_clone_id"],
    )
    logger.info("building AIRR lookup structures")
    records_by_clone = {
        row[args.clone_id_col]: {"junction_aa": row["junction_aa"]}
        for _, row in airr_df.iterrows()
        if args.clone_id_col in airr_df.columns
    }
    if args.clone_id_col in airr_df.columns:
        logger.info("prepared AIRR clone_id lookup entries=%d", len(records_by_clone))

    if args.clone_id_col not in airr_df.columns:
        if len(airr_df) != embedding_info["num_rows"]:
            raise ValueError(
                f"AIRR table has {len(airr_df)} rows but embeddings file has {embedding_info['num_rows']} rows. "
                "Row-order alignment is only supported when lengths match."
            )
        alignment_mode = "row_order"
        logger.info("using row-order alignment")
        matched_keys = list(range(len(airr_df)))
        records_by_key = {idx: {"junction_aa": seq} for idx, seq in enumerate(airr_df["junction_aa"].tolist())}
    else:
        alignment_mode = "clone_id"
        logger.info(
            "using clone_id alignment with %d AIRR clone_ids; scanning embeddings parquet for matches",
            len(records_by_clone),
        )
        matched_key_set = set()
        scanned_rows = 0
        scanned_batches = 0
        for clone_ids, emb_batch in iter_embedding_batches(
            args.embeddings_path,
            batch_size=args.reader_batch_size,
            clone_id_col=args.clone_id_col,
            embedding_column=args.embedding_column,
            include_clone_id=True,
        ):
            scanned_batches += 1
            scanned_rows += len(emb_batch)
            for clone_id in clone_ids:
                if clone_id in records_by_clone:
                    if clone_id in matched_key_set:
                        raise ValueError(
                            f"Embeddings table contains duplicate {args.clone_id_col} values among AIRR-matched rows."
                        )
                    matched_key_set.add(clone_id)

            if scanned_batches % 50 == 0:
                logger.info(
                    "embedding scan progress batches=%d rows=%d matched_airr_rows=%d",
                    scanned_batches,
                    scanned_rows,
                    len(matched_key_set),
                )

        matched_keys = [clone_id for clone_id in records_by_clone if clone_id in matched_key_set]
        if not matched_keys:
            raise ValueError(f"No rows matched between AIRR and embeddings tables by {args.clone_id_col}.")
        records_by_key = records_by_clone
        logger.info(
            "finished embeddings scan batches=%d rows=%d matched_airr_rows=%d",
            scanned_batches,
            scanned_rows,
            len(matched_keys),
        )

    matched_key_set = set(matched_keys)
    logger.info("building matched AIRR dataframe")
    if alignment_mode == "row_order":
        matched_df = airr_df.copy().reset_index(drop=True)
    else:
        matched_df = airr_df[airr_df[args.clone_id_col].isin(matched_key_set)].reset_index(drop=True)

    logger.info("splitting matched keys into train/val/test")
    train_idx, val_idx, test_idx = split_indices(
        len(matched_keys),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    split_keys = {
        "train": {matched_keys[idx] for idx in train_idx},
        "val": {matched_keys[idx] for idx in val_idx},
        "test": {matched_keys[idx] for idx in test_idx},
    }
    merge_stats = {
        "airr_rows": int(len(airr_df)),
        "embeddings_rows": int(embedding_info["num_rows"]),
        "merged_rows": int(len(matched_keys)),
        "airr_unmatched_rows": int(len(airr_df) - len(matched_keys)),
        "embeddings_unmatched_rows": int(embedding_info["num_rows"] - len(matched_keys)),
        "clone_id_column": args.clone_id_col,
        "embedding_column": args.embedding_column,
        "alignment_mode": alignment_mode,
        "embedding_dim": int(embedding_info["embedding_dim"]),
    }
    logger.info(
        "prepare_streaming_splits: done merged_rows=%d train=%d val=%d test=%d",
        len(matched_keys),
        len(split_keys["train"]),
        len(split_keys["val"]),
        len(split_keys["test"]),
    )
    return matched_df, records_by_key, split_keys, merge_stats, build_embedding_iterator(args, alignment_mode)


def fit_streaming_standardizer_with_logging(iter_batches_fn, selected_keys, embedding_dim, logger=None):
    logger = logger or logging.getLogger("irrm_codec")
    logger.info(
        "fitting embedding standardizer on train split rows=%d embedding_dim=%d",
        len(selected_keys),
        embedding_dim,
    )
    count = 0
    feature_sum = np.zeros(embedding_dim, dtype=np.float64)
    feature_sum_sq = np.zeros(embedding_dim, dtype=np.float64)
    scanned_batches = 0
    scanned_rows = 0

    for keys, emb_batch in iter_batches_fn():
        scanned_batches += 1
        scanned_rows += len(emb_batch)
        row_indices = [row_idx for row_idx, key in enumerate(keys) if key in selected_keys]
        if not row_indices:
            if scanned_batches % 50 == 0:
                logger.info(
                    "standardizer progress batches=%d scanned_rows=%d train_rows_collected=%d",
                    scanned_batches,
                    scanned_rows,
                    count,
                )
            continue

        embeddings = np.asarray(emb_batch[row_indices], dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embedding batch, got shape {embeddings.shape}.")
        if embeddings.shape[1] != embedding_dim:
            raise ValueError(f"Expected embedding dimension {embedding_dim}, got {embeddings.shape[1]}.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings matrix contains NaN or infinite values.")

        feature_sum += embeddings.sum(axis=0, dtype=np.float64)
        feature_sum_sq += np.square(embeddings, dtype=np.float64).sum(axis=0)
        count += embeddings.shape[0]

        if scanned_batches % 50 == 0:
            logger.info(
                "standardizer progress batches=%d scanned_rows=%d train_rows_collected=%d",
                scanned_batches,
                scanned_rows,
                count,
            )

    if count == 0:
        raise ValueError("No training embeddings were found while fitting the standardizer.")
    if count != len(selected_keys):
        raise ValueError(
            f"Expected {len(selected_keys)} training embeddings while fitting the standardizer, found {count}."
        )

    mean = feature_sum / count
    variance = np.maximum(feature_sum_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-8, 1.0, std)
    logger.info(
        "finished standardizer fit batches=%d scanned_rows=%d train_rows=%d",
        scanned_batches,
        scanned_rows,
        count,
    )
    return mean.astype(np.float32), std.astype(np.float32)


def build_streaming_dataloader(
    *,
    task,
    collate_fn,
    records_by_key,
    selected_keys,
    iter_batches_fn,
    batch_size,
    max_len,
    shuffle,
    num_workers,
    mean,
    std,
    seed,
):
    logger = logging.getLogger("irrm_codec")
    logger.info(
        "creating streaming dataloader task=%s rows=%d batch_size=%d shuffle=%s num_workers=%d",
        task,
        len(selected_keys),
        batch_size,
        shuffle,
        num_workers,
    )
    dataset = StreamingEmbeddingDataset(
        task=task,
        records_by_key=records_by_key,
        selected_keys=selected_keys,
        iter_embedding_batches_fn=iter_batches_fn,
        max_len=max_len,
        mean=mean,
        std=std,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
