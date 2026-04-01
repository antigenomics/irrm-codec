import json
import logging
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_standardizer(x):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_standardizer_on_indices(x, indices, chunk_size):
    x = np.asarray(x, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}.")
    if len(indices) == 0:
        raise ValueError("Cannot fit standardizer on an empty index set.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    sum_x = np.zeros(x.shape[1], dtype=np.float64)
    sum_x2 = np.zeros(x.shape[1], dtype=np.float64)
    total = 0

    for start in range(0, len(indices), chunk_size):
        batch_idx = indices[start : start + chunk_size]
        chunk = x[batch_idx].astype(np.float32, copy=False)
        sum_x += chunk.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(chunk, dtype=np.float32).sum(axis=0, dtype=np.float64)
        total += len(batch_idx)

    mean = sum_x / total
    variance = np.maximum(sum_x2 / total - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(x, mean, std):
    x = np.asarray(x, dtype=np.float32)
    return (x - mean) / std


def create_temp_train_shard_dir(prefix="irrm_codec_train_"):
    return tempfile.TemporaryDirectory(prefix=prefix)


def load_train_shard(shard_info):
    sequences = np.load(Path(shard_info["seq_path"]), allow_pickle=False)
    embeddings = np.load(Path(shard_info["emb_path"]), mmap_mode="r")
    return sequences.tolist(), embeddings


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_indices(num_items, train_fraction=0.8, val_fraction=0.1, seed=42):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_items)
    train_end = int(num_items * train_fraction)
    val_end = train_end + int(num_items * val_fraction)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def move_to_device(items, device):
    return [item.to(device) if hasattr(item, "to") else item for item in items]


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(path, model, optimizer, epoch, metrics, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "extra": extra or {},
    }
    torch.save(payload, path)


def summarize_metrics(metric_sums, steps):
    if steps == 0:
        return {}
    return {name: value / steps for name, value in metric_sums.items()}


def setup_logging(log_path=None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("irrm_codec")
