import json
import logging
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch

from irrm_codec.dataio import iter_embedding_batches, read_airr_table
from irrm_codec.datasets import CachedBatchDataset
from irrm_codec.utils import save_json, split_indices


def _resolve_cache_dir(args):
    base_dir = Path(getattr(args, "cache_dir", "") or (Path(args.output_dir) / "batch_cache"))
    return base_dir / f"run_{uuid.uuid4().hex}"


def prepare_batch_cache(args, logger=None):
    logger = logger or logging.getLogger("irrm_codec")
    cache_dir = _resolve_cache_dir(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    airr_columns = ["junction_aa"]
    if args.clone_id_col:
        airr_columns.append(args.clone_id_col)
    if args.locus is not None:
        airr_columns.append("locus")

    logger.info("creating batch cache in %s", cache_dir)
    logger.info("reading minimal AIRR columns from %s", args.airr_path)
    airr_df = read_airr_table(
        args.airr_path,
        clone_id_col=args.clone_id_col,
        columns=list(dict.fromkeys(airr_columns)),
        validate=False,
    )
    if args.locus is not None and "locus" in airr_df.columns:
        logger.info("filtering AIRR by locus=%s", args.locus)
        airr_df = airr_df[airr_df["locus"] == args.locus].reset_index(drop=True)
    logger.info("AIRR rows after filtering=%d", len(airr_df))

    seqs = airr_df["junction_aa"].astype(str).to_numpy(copy=True)
    num_rows = len(seqs)
    if num_rows == 0:
        raise ValueError("AIRR table is empty after filtering.")

    logger.info("creating train/val/test split")
    split_names = ("train", "val", "test")
    split_ids = np.empty(num_rows, dtype=np.int8)
    train_idx, val_idx, test_idx = split_indices(
        num_rows,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    split_ids[train_idx] = 0
    split_ids[val_idx] = 1
    split_ids[test_idx] = 2
    split_row_counts = {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
    }
    logger.info(
        "split ready train=%d val=%d test=%d",
        split_row_counts["train"],
        split_row_counts["val"],
        split_row_counts["test"],
    )

    if args.clone_id_col in airr_df.columns:
        logger.info("building lightweight clone_id -> row_index map")
        row_index_by_clone_id = {clone_id: idx for idx, clone_id in enumerate(airr_df[args.clone_id_col].tolist())}
        include_clone_id = True
        alignment_mode = "clone_id"
        logger.info("clone_id map ready entries=%d", len(row_index_by_clone_id))
    else:
        row_index_by_clone_id = None
        include_clone_id = False
        alignment_mode = "row_order"
        logger.info("using row-order alignment")

    cache_batch_size = int(getattr(args, "cache_batch_size", 4096))
    logger.info(
        "starting cache population reader_batch_size=%d cache_batch_size=%d alignment_mode=%s",
        args.reader_batch_size,
        cache_batch_size,
        alignment_mode,
    )

    buffers = {name: {"seqs": [], "embeddings": []} for name in split_names}
    shard_counts = {name: 0 for name in split_names}
    shard_entries = {name: [] for name in split_names}

    def flush(split_name):
        buffer = buffers[split_name]
        row_count = len(buffer["seqs"])
        if row_count == 0:
            return
        shard_path = cache_dir / f"{split_name}_{shard_counts[split_name]:06d}.npz"
        logger.info("writing shard split=%s rows=%d path=%s", split_name, row_count, shard_path.name)
        seq_array = np.asarray(buffer["seqs"])
        embedding_array = np.asarray(buffer["embeddings"], dtype=np.float32)
        np.savez_compressed(
            shard_path,
            seqs=seq_array,
            embeddings=embedding_array,
        )
        shard_entries[split_name].append({"path": shard_path.name, "rows": row_count})
        shard_counts[split_name] += 1
        buffer["seqs"].clear()
        buffer["embeddings"].clear()
        del seq_array
        del embedding_array

    embedding_dim = None
    train_sum = None
    train_sum_sq = None
    train_count = 0
    matched_rows = 0
    scanned_rows = 0
    scanned_batches = 0
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

        if embedding_dim is None:
            embedding_dim = int(emb_batch.shape[1])
            train_sum = np.zeros(embedding_dim, dtype=np.float64)
            train_sum_sq = np.zeros(embedding_dim, dtype=np.float64)
            logger.info("detected embedding_dim=%d from first batch", embedding_dim)

        if alignment_mode == "row_order":
            limit = min(len(emb_batch), num_rows - row_offset)
            for local_idx in range(limit):
                row_idx = row_offset
                split_id = int(split_ids[row_idx])
                split_name = split_names[split_id]
                emb = emb_batch[local_idx].astype(np.float32, copy=False)
                buffers[split_name]["seqs"].append(seqs[row_idx])
                buffers[split_name]["embeddings"].append(emb)
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
                row_idx = row_index_by_clone_id.get(clone_id)
                if row_idx is None:
                    continue
                split_id = int(split_ids[row_idx])
                split_name = split_names[split_id]
                emb = emb_batch[local_idx].astype(np.float32, copy=False)
                buffers[split_name]["seqs"].append(seqs[row_idx])
                buffers[split_name]["embeddings"].append(emb)
                matched_rows += 1
                if split_id == 0:
                    train_sum += emb
                    train_sum_sq += np.square(emb, dtype=np.float64)
                    train_count += 1
                if len(buffers[split_name]["seqs"]) >= cache_batch_size:
                    flush(split_name)

        if scanned_batches == 1 or scanned_batches % 10 == 0:
            logger.info(
                "cache progress batches=%d scanned_rows=%d matched_rows=%d train_rows=%d",
                scanned_batches,
                scanned_rows,
                matched_rows,
                train_count,
            )
        del emb_batch
        if clone_ids is not None:
            del clone_ids

    for split_name in split_names:
        flush(split_name)

    if embedding_dim is None or train_count == 0:
        raise ValueError("No training embeddings were cached.")

    logger.info("computing train standardizer")
    mean = (train_sum / train_count).astype(np.float32)
    variance = np.maximum(train_sum_sq / train_count - np.square(mean, dtype=np.float64), 0.0)
    std = np.sqrt(variance).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    np.save(cache_dir / "mean.npy", mean)
    np.save(cache_dir / "std.npy", std)
    logger.info("saved mean.npy and std.npy")

    merge_stats = {
        "airr_rows": int(num_rows),
        "embeddings_rows": int(scanned_rows),
        "merged_rows": int(matched_rows),
        "airr_unmatched_rows": int(max(num_rows - matched_rows, 0)),
        "embeddings_unmatched_rows": int(max(scanned_rows - matched_rows, 0)),
        "clone_id_column": args.clone_id_col,
        "embedding_column": args.embedding_column,
        "alignment_mode": alignment_mode,
        "embedding_dim": int(embedding_dim),
    }
    data_stats = {
        "num_samples": int(num_rows),
        "embedding_dim": int(embedding_dim),
        "max_len": int(args.max_len),
    }
    manifest = {
        "cache_version": 1,
        "cache_dir": str(cache_dir),
        "cache_batch_size": cache_batch_size,
        "reader_batch_size": int(args.reader_batch_size),
        "standardizer": {"mean_path": "mean.npy", "std_path": "std.npy"},
        "splits": shard_entries,
        "split_row_counts": split_row_counts,
        "data_stats": data_stats,
        "merge_stats": merge_stats,
        "airr_path": args.airr_path,
        "embeddings_path": args.embeddings_path,
    }
    logger.info("writing cache manifest path=%s", manifest_path)
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
    split_row_counts = manifest["split_row_counts"]
    cache_dir = Path(manifest["cache_dir"])

    shard_paths = {
        split_name: [cache_dir / item["path"] for item in manifest["splits"][split_name]]
        for split_name in ("train", "val", "test")
    }

    logger.info("building dataloaders")
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

    return {
        "manifest": manifest,
        "mean": mean,
        "std": std,
        "data_stats": manifest["data_stats"],
        "merge_stats": manifest["merge_stats"],
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
