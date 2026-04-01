import json
import logging
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch

from irrm_codec.dataio import inspect_embeddings_file, iter_embedding_batches, read_airr_table
from irrm_codec.datasets import CachedBatchDataset, validate_airr_dataframe
from irrm_codec.utils import save_json, split_indices


def _resolve_cache_dir(args):
    base_dir = Path(getattr(args, "cache_dir", "") or (Path(args.output_dir) / "batch_cache"))
    return base_dir / f"run_{uuid.uuid4().hex}"


def prepare_batch_cache(args, logger=None):
    logger = logger or logging.getLogger("irrm_codec")
    cache_dir = _resolve_cache_dir(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    logger.info("creating batch cache in %s", cache_dir)
    logger.info("reading AIRR table from %s", args.airr_path)
    airr_df = read_airr_table(args.airr_path, clone_id_col=args.clone_id_col)
    if args.locus is not None:
        logger.info("filtering AIRR table by locus=%s", args.locus)
        airr_df = airr_df[airr_df["locus"] == args.locus].reset_index(drop=True)
    if len(airr_df) == 0:
        raise ValueError("AIRR table is empty after locus filtering.")
    logger.info("AIRR rows after filtering=%d", len(airr_df))

    data_stats = validate_airr_dataframe(airr_df, max_len=args.max_len)
    embedding_info = inspect_embeddings_file(
        args.embeddings_path,
        clone_id_col=args.clone_id_col,
        embedding_column=args.embedding_column,
    )
    logger.info(
        "embeddings metadata rows=%d embedding_dim=%d has_clone_id=%s",
        embedding_info["num_rows"],
        embedding_info["embedding_dim"],
        embedding_info["has_clone_id"],
    )

    split_names = ("train", "val", "test")
    split_indices_array = np.empty(len(airr_df), dtype=np.int8)
    train_idx, val_idx, test_idx = split_indices(
        len(airr_df),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    split_indices_array[train_idx] = 0
    split_indices_array[val_idx] = 1
    split_indices_array[test_idx] = 2

    buffers = {name: {"seqs": [], "embeddings": []} for name in split_names}
    shard_counts = {name: 0 for name in split_names}
    shard_entries = {name: [] for name in split_names}
    split_row_counts = {name: 0 for name in split_names}

    def flush(split_name):
        buffer = buffers[split_name]
        row_count = len(buffer["seqs"])
        if row_count == 0:
            return
        shard_path = cache_dir / f"{split_name}_{shard_counts[split_name]:06d}.npz"
        np.savez_compressed(
            shard_path,
            seqs=np.asarray(buffer["seqs"]),
            embeddings=np.asarray(buffer["embeddings"], dtype=np.float32),
        )
        shard_entries[split_name].append({"path": shard_path.name, "rows": row_count})
        shard_counts[split_name] += 1
        buffer["seqs"].clear()
        buffer["embeddings"].clear()

    embedding_dim = embedding_info["embedding_dim"]
    train_sum = np.zeros(embedding_dim, dtype=np.float64)
    train_sum_sq = np.zeros(embedding_dim, dtype=np.float64)
    train_count = 0
    matched_rows = 0
    scanned_rows = 0
    scanned_batches = 0

    if args.clone_id_col in airr_df.columns:
        logger.info("building AIRR clone_id lookup")
        airr_lookup = {
            row[args.clone_id_col]: (row["junction_aa"], int(split_indices_array[idx]))
            for idx, (_, row) in enumerate(airr_df.iterrows())
        }
        logger.info("AIRR lookup ready entries=%d", len(airr_lookup))
        include_clone_id = True
        alignment_mode = "clone_id"
    else:
        airr_lookup = None
        include_clone_id = False
        alignment_mode = "row_order"
        logger.info("using row-order alignment for cache creation")

    cache_batch_size = int(getattr(args, "cache_batch_size", 4096))
    logger.info(
        "starting cache build reader_batch_size=%d cache_batch_size=%d alignment_mode=%s",
        args.reader_batch_size,
        cache_batch_size,
        alignment_mode,
    )

    row_offset = 0
    for clone_ids, emb_batch in iter_embedding_batches(
        args.embeddings_path,
        batch_size=args.reader_batch_size,
        clone_id_col=args.clone_id_col,
        embedding_column=args.embedding_column,
        include_clone_id=include_clone_id,
    ):
        scanned_batches += 1
        scanned_rows += len(emb_batch)
        if emb_batch.shape[1] != embedding_dim:
            raise ValueError(f"Expected embedding dimension {embedding_dim}, got {emb_batch.shape[1]}.")

        if alignment_mode == "row_order":
            limit = min(len(emb_batch), len(airr_df) - row_offset)
            for local_idx in range(limit):
                split_id = int(split_indices_array[row_offset])
                split_name = split_names[split_id]
                seq = airr_df.iloc[row_offset]["junction_aa"]
                emb = np.asarray(emb_batch[local_idx], dtype=np.float32)
                buffers[split_name]["seqs"].append(seq)
                buffers[split_name]["embeddings"].append(emb)
                split_row_counts[split_name] += 1
                matched_rows += 1
                if split_id == 0:
                    train_sum += emb
                    train_sum_sq += np.square(emb, dtype=np.float64)
                    train_count += 1
                if len(buffers[split_name]["seqs"]) >= cache_batch_size:
                    flush(split_name)
                row_offset += 1
        else:
            for local_idx, clone_id in enumerate(clone_ids):
                record = airr_lookup.get(clone_id)
                if record is None:
                    continue
                seq, split_id = record
                split_name = split_names[split_id]
                emb = np.asarray(emb_batch[local_idx], dtype=np.float32)
                buffers[split_name]["seqs"].append(seq)
                buffers[split_name]["embeddings"].append(emb)
                split_row_counts[split_name] += 1
                matched_rows += 1
                if split_id == 0:
                    train_sum += emb
                    train_sum_sq += np.square(emb, dtype=np.float64)
                    train_count += 1
                if len(buffers[split_name]["seqs"]) >= cache_batch_size:
                    flush(split_name)

        if scanned_batches % 10 == 0:
            logger.info(
                "cache build progress batches=%d scanned_rows=%d matched_rows=%d train_rows=%d",
                scanned_batches,
                scanned_rows,
                matched_rows,
                train_count,
            )

    for split_name in split_names:
        flush(split_name)

    if train_count == 0:
        raise ValueError("No training rows were cached.")

    mean = (train_sum / train_count).astype(np.float32)
    variance = np.maximum(train_sum_sq / train_count - np.square(mean, dtype=np.float64), 0.0)
    std = np.sqrt(variance).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    np.save(cache_dir / "mean.npy", mean)
    np.save(cache_dir / "std.npy", std)

    merge_stats = {
        "airr_rows": int(len(airr_df)),
        "embeddings_rows": int(embedding_info["num_rows"]),
        "merged_rows": int(matched_rows),
        "airr_unmatched_rows": int(len(airr_df) - matched_rows if alignment_mode == "row_order" else len(airr_df) - matched_rows),
        "embeddings_unmatched_rows": int(embedding_info["num_rows"] - matched_rows),
        "clone_id_column": args.clone_id_col,
        "embedding_column": args.embedding_column,
        "alignment_mode": alignment_mode,
        "embedding_dim": int(embedding_dim),
    }

    manifest = {
        "cache_version": 1,
        "cache_dir": str(cache_dir),
        "cache_batch_size": cache_batch_size,
        "reader_batch_size": int(args.reader_batch_size),
        "standardizer": {"mean_path": "mean.npy", "std_path": "std.npy"},
        "splits": shard_entries,
        "split_row_counts": split_row_counts,
        "data_stats": {**data_stats, "embedding_dim": int(embedding_dim)},
        "merge_stats": merge_stats,
        "airr_path": args.airr_path,
        "embeddings_path": args.embeddings_path,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "cache build done train_shards=%d val_shards=%d test_shards=%d",
        len(shard_entries["train"]),
        len(shard_entries["val"]),
        len(shard_entries["test"]),
    )
    return manifest, mean, std


def cleanup_batch_cache(cache_dir, logger=None):
    logger = logger or logging.getLogger("irrm_codec")
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return
    logger.info("removing batch cache directory %s", cache_dir)
    shutil.rmtree(cache_dir, ignore_errors=True)


def build_cached_dataloader(
    *,
    task,
    collate_fn,
    shard_paths,
    num_rows,
    batch_size,
    max_len,
    shuffle,
    num_workers,
    mean,
    std,
    seed,
):
    dataset = CachedBatchDataset(
        task=task,
        shard_paths=shard_paths,
        max_len=max_len,
        mean=mean,
        std=std,
        shuffle=shuffle,
        seed=seed,
        num_rows=num_rows,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def prepare_cached_training_data(args, logger, *, task, collate_fn):
    manifest, mean, std = prepare_batch_cache(args, logger=logger)
    data_stats = manifest["data_stats"]
    merge_stats = manifest["merge_stats"]
    split_row_counts = manifest["split_row_counts"]
    cache_dir = Path(manifest["cache_dir"])

    shard_paths = {
        split_name: [cache_dir / item["path"] for item in manifest["splits"][split_name]]
        for split_name in ("train", "val", "test")
    }

    logger.info("building dataloaders")
    logger.info("creating train dataloader")
    train_loader = build_cached_dataloader(
        task=task,
        collate_fn=collate_fn,
        shard_paths=shard_paths["train"],
        num_rows=split_row_counts["train"],
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=True,
        num_workers=args.num_workers,
        mean=mean,
        std=std,
        seed=args.seed,
    )
    logger.info("creating val dataloader")
    val_loader = build_cached_dataloader(
        task=task,
        collate_fn=collate_fn,
        shard_paths=shard_paths["val"],
        num_rows=split_row_counts["val"],
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=False,
        num_workers=args.num_workers,
        mean=mean,
        std=std,
        seed=args.seed,
    )
    logger.info("creating test dataloader")
    test_loader = build_cached_dataloader(
        task=task,
        collate_fn=collate_fn,
        shard_paths=shard_paths["test"],
        num_rows=split_row_counts["test"],
        batch_size=args.batch_size,
        max_len=args.max_len,
        shuffle=False,
        num_workers=args.num_workers,
        mean=mean,
        std=std,
        seed=args.seed,
    )
    logger.info("dataloaders ready; starting model initialization")

    return {
        "manifest": manifest,
        "mean": mean,
        "std": std,
        "data_stats": data_stats,
        "merge_stats": merge_stats,
        "split_row_counts": split_row_counts,
        "cache_dir": cache_dir,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def save_training_metadata(output_dir, args, data_stats, merge_stats, split_row_counts, manifest):
    save_json(
        Path(output_dir) / "data_stats.json",
        {
            **data_stats,
            **merge_stats,
            "airr_path": args.airr_path,
            "embeddings_path": args.embeddings_path,
            "train_size": int(split_row_counts["train"]),
            "val_size": int(split_row_counts["val"]),
            "test_size": int(split_row_counts["test"]),
            "standardizer": manifest["standardizer"],
            "checkpoints": {"best": "best.pt", "last": "last.pt"},
        },
    )
