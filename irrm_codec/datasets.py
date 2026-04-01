import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from irrm_codec.tokenization import BOS_ID, EOS_ID, PAD_ID, UNK_ID, encode


def validate_airr_dataframe(df, max_len=40):
    required_columns = {"junction_aa", "v_call", "j_call", "locus"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataframe is missing required columns: {missing}")

    if len(df) == 0:
        raise ValueError("Dataframe is empty after filtering.")

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
        "num_unique_clone_ids": int(df["clone_id"].nunique()) if "clone_id" in df.columns else int(len(df)),
        "min_length": int(min(sequence_lengths)),
        "max_length": int(max(sequence_lengths)),
        "mean_length": float(np.mean(sequence_lengths)),
        "truncated_fraction": truncated_sequences / len(df),
        "unk_sequence_fraction": unk_sequences / len(df),
        "max_len": max_len,
    }


class CachedBatchDataset(IterableDataset):
    def __init__(self, *, task, shard_paths, max_len, mean, std, shuffle=False, seed=42, num_rows=None):
        self.task = task
        self.shard_paths = [str(path) for path in shard_paths]
        self.max_len = max_len
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_rows = num_rows

    def __len__(self):
        if self.num_rows is None:
            total = 0
            for shard_path in self.shard_paths:
                with np.load(shard_path) as payload:
                    total += len(payload["seqs"])
            self.num_rows = total
        return self.num_rows

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _make_item(self, seq, embedding):
        tokens = encode(seq, self.max_len)
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        embedding_tensor = torch.from_numpy(embedding)

        if self.task == "forward":
            return {
                "tokens": token_tensor,
                "embedding": embedding_tensor,
                "length": len(tokens),
            }

        return {
            "embedding": embedding_tensor,
            "decoder_input": torch.cat([torch.tensor([BOS_ID], dtype=torch.long), token_tensor], dim=0),
            "target": torch.cat([token_tensor, torch.tensor([EOS_ID], dtype=torch.long)], dim=0),
            "length": len(tokens),
        }

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = np.random.default_rng(self.seed + self.epoch + worker_id)

        shard_indices = np.arange(len(self.shard_paths))
        if self.shuffle and len(shard_indices) > 1:
            rng.shuffle(shard_indices)

        for position, shard_idx in enumerate(shard_indices):
            if position % num_workers != worker_id:
                continue

            with np.load(self.shard_paths[int(shard_idx)]) as payload:
                seqs = payload["seqs"]
                embeddings = payload["embeddings"].astype(np.float32, copy=False)

            row_indices = np.arange(len(seqs))
            if self.shuffle and len(row_indices) > 1:
                rng.shuffle(row_indices)

            standardized = ((embeddings[row_indices] - self.mean) / self.std).astype(np.float32, copy=False)
            for seq, embedding in zip(seqs[row_indices], standardized):
                yield self._make_item(str(seq), embedding)


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
