import json
import random
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


def apply_standardizer(x, mean, std):
    x = np.asarray(x, dtype=np.float32)
    return (x - mean) / std


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
