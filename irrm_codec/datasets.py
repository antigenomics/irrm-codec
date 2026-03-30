import numpy as np
import torch
from torch.utils.data import Dataset

from irrm_codec.tokenization import BOS_ID, EOS_ID, PAD_ID, UNK_ID, encode


def validate_dataframe(df, emb_array, max_len=40, emb_dim=9000):
    required_columns = {"clone_id", "junction_aa", "v_call", "j_call", "locus"}
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
    if emb_array.shape[1] != emb_dim:
        raise ValueError(f"Expected embedding dimension {emb_dim}, got {emb_array.shape[1]}.")
    if not np.isfinite(emb_array).all():
        raise ValueError("Embedding array contains NaN or infinite values.")

    sequence_lengths = []
    unk_sequences = 0
    truncated_sequences = 0
    empty_sequences = 0

    for raw_seq in df["junction_aa"].tolist():
        seq = "" if raw_seq is None else str(raw_seq).strip().upper()
        if not seq:
            empty_sequences += 1
            continue
        sequence_lengths.append(len(seq))
        unk_sequences += int(any(char not in "ACDEFGHIKLMNPQRSTVWY" for char in seq))
        truncated_sequences += int(len(seq) > max_len)

    if empty_sequences:
        raise ValueError(f"Found {empty_sequences} empty or missing sequences.")

    return {
        "num_samples": len(df),
        "embedding_dim": emb_array.shape[1],
        "num_unique_clone_ids": int(df["clone_id"].nunique()),
        "min_length": int(min(sequence_lengths)),
        "max_length": int(max(sequence_lengths)),
        "mean_length": float(np.mean(sequence_lengths)),
        "truncated_fraction": truncated_sequences / len(df),
        "unk_sequence_fraction": unk_sequences / len(df),
        "max_len": max_len,
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
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        return {
            "embedding": torch.from_numpy(self.embs[idx]),
            "decoder_input": torch.cat(
                [torch.tensor([BOS_ID], dtype=torch.long), token_tensor], dim=0
            ),
            "target": torch.cat(
                [token_tensor, torch.tensor([EOS_ID], dtype=torch.long)], dim=0
            ),
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
    decoder_input = torch.nn.utils.rnn.pad_sequence(
        [item["decoder_input"] for item in batch],
        batch_first=True,
        padding_value=PAD_ID,
    )
    target = torch.nn.utils.rnn.pad_sequence(
        [item["target"] for item in batch],
        batch_first=True,
        padding_value=PAD_ID,
    )
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    target_mask = target.ne(PAD_ID)
    unk_fraction = target.eq(UNK_ID).logical_and(target_mask).float().sum() / target_mask.float().sum()
    return emb, decoder_input, target, lengths, unk_fraction
