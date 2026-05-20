import numpy as np
import torch
from torch.utils.data import Dataset

from irrm_codec.tokenization import PAD_ID, UNK_ID, VALID_AA, encode, gap_pad_cdr3


def validate_dataframe(df, emb_array, max_len=40, clone_id_col="clone_id"):
    required_columns = {"junction_aa", "v_call", "j_call", "locus"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataframe is missing required columns: {missing}")

    if len(df) == 0:
        raise ValueError("Dataframe is empty after filtering.")

    emb_array = np.asarray(emb_array, dtype=np.float32)
    if emb_array.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {emb_array.shape}.")
    if emb_array.shape[0] != len(df):
        raise ValueError(
            f"Embedding count {emb_array.shape[0]} does not match dataframe length {len(df)}."
        )
    if not np.isfinite(emb_array).all():
        raise ValueError("Embedding array contains NaN or infinite values.")

    sequence_lengths = []
    processed_lengths = []
    unk_sequences = 0
    truncated_sequences = 0
    empty_sequences = 0
    overlength_sequences = 0

    for raw_seq in df["junction_aa"].tolist():
        seq = "" if raw_seq is None else str(raw_seq).strip().upper()
        if not seq:
            empty_sequences += 1
            continue
        sequence_lengths.append(len(seq))
        unk_sequences += int(any(char not in VALID_AA for char in seq))
        try:
            processed_seq = gap_pad_cdr3(seq, target_len=max_len)
        except ValueError as exc:
            if "exceeds target length" in str(exc):
                overlength_sequences += 1
                truncated_sequences += 1
                continue
            raise
        processed_lengths.append(len(processed_seq))

    if empty_sequences:
        raise ValueError(f"Found {empty_sequences} empty or missing sequences.")
    if overlength_sequences:
        raise ValueError(
            f"Found {overlength_sequences} sequences longer than target length {max_len}; "
            "they cannot be converted to fixed length by inserting gaps."
        )

    return {
        "num_samples": len(df),
        "embedding_dim": emb_array.shape[1],
        "num_unique_clone_ids": (
            int(df[clone_id_col].nunique()) if clone_id_col in df.columns else int(len(df))
        ),
        "min_length": int(min(sequence_lengths)),
        "max_length": int(max(sequence_lengths)),
        "mean_length": float(np.mean(sequence_lengths)),
        "processed_min_length": int(min(processed_lengths)),
        "processed_max_length": int(max(processed_lengths)),
        "processed_mean_length": float(np.mean(processed_lengths)),
        "truncated_fraction": truncated_sequences / len(df),
        "unk_sequence_fraction": unk_sequences / len(df),
        "max_len": max_len,
    }


def validate_target_dataframe(df, target_array, max_len=40, clone_id_col="clone_id", target_name="target"):
    target_array = np.asarray(target_array, dtype=np.float32)
    if target_array.ndim != 1:
        raise ValueError(f"Expected 1D target array, got shape {target_array.shape}.")
    if target_array.shape[0] != len(df):
        raise ValueError(
            f"Target count {target_array.shape[0]} does not match dataframe length {len(df)}."
        )
    if not np.isfinite(target_array).all():
        raise ValueError(f"Target array {target_name!r} contains NaN or infinite values.")

    # Reuse sequence validation from the forward path with a lightweight dummy array.
    base_stats = validate_dataframe(
        df,
        np.zeros((len(df), 1), dtype=np.float32),
        max_len=max_len,
        clone_id_col=clone_id_col,
    )
    return {
        **base_stats,
        "target_name": target_name,
        "target_min": float(np.min(target_array)),
        "target_max": float(np.max(target_array)),
        "target_mean": float(np.mean(target_array)),
        "target_std": float(np.std(target_array)),
    }


class ForwardDataset(Dataset):
    def __init__(self, df, emb_array, max_len=40):
        self.seqs = df["junction_aa"].tolist()
        self.embs = np.asarray(emb_array, dtype=np.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = encode(self.seqs[idx], self.max_len)
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "embedding": torch.from_numpy(self.embs[idx]),
            "length": len(tokens),
        }


class InverseDataset(Dataset):
    def __init__(self, df, emb_array, max_len=40):
        self.seqs = df["junction_aa"].tolist()
        self.embs = np.asarray(emb_array, dtype=np.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = encode(self.seqs[idx], self.max_len)
        target = torch.tensor(tokens, dtype=torch.long)
        return {
            "embedding": torch.from_numpy(self.embs[idx]),
            "target": target,
        }


class PgenDataset(Dataset):
    def __init__(self, df, target_array, max_len=40):
        self.seqs = df["junction_aa"].tolist()
        self.targets = np.asarray(target_array, dtype=np.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = encode(self.seqs[idx], self.max_len)
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "length": len(tokens),
        }


def collate_forward(batch):
    tokens = torch.nn.utils.rnn.pad_sequence(
        [item["tokens"] for item in batch],
        batch_first=True,
        padding_value=PAD_ID,
    )
    emb = torch.stack([item["embedding"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    mask = tokens.ne(PAD_ID)
    return tokens, mask, emb, lengths


def collate_inverse(batch):
    emb = torch.stack([item["embedding"] for item in batch])
    target = torch.stack([item["target"] for item in batch])
    target_mask = target.ne(PAD_ID)
    unk_fraction = target.eq(UNK_ID).logical_and(target_mask).float().sum() / target_mask.float().sum()
    return emb, target, unk_fraction


def collate_pgen(batch):
    tokens = torch.nn.utils.rnn.pad_sequence(
        [item["tokens"] for item in batch],
        batch_first=True,
        padding_value=PAD_ID,
    )
    targets = torch.stack([item["target"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    mask = tokens.ne(PAD_ID)
    return tokens, mask, targets, lengths
