import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from irrm_codec.dataio import load_airr_with_embeddings
from irrm_codec.datasets import ForwardDataset, collate_forward, validate_dataframe
from irrm_codec.forward_model import ForwardModel
from irrm_codec.losses import forward_loss, forward_metrics
from irrm_codec.utils import (
    apply_standardizer,
    choose_device,
    fit_standardizer,
    move_to_device,
    save_checkpoint,
    save_json,
    set_seed,
    split_indices,
    summarize_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the forward IRRM-CODEC model.")
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--output-dir", default="artifacts/forward")
    parser.add_argument("--locus", default="alpha")
    parser.add_argument("--clone-id-col", default="clone_id")
    parser.add_argument("--embedding-column", default="tcremp_emb")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_dataloader(df, emb, batch_size, max_len, shuffle, num_workers):
    dataset = ForwardDataset(df, emb, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_forward,
    )


def run_epoch(model, loader, optimizer, device):
    is_train = optimizer is not None
    model.train(mode=is_train)

    metric_sums = {"loss": 0.0, "mse": 0.0, "cosine": 0.0}
    steps = 0

    for batch in loader:
        tokens, mask, target, _lengths = move_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            pred = model(tokens, mask)
            loss = forward_loss(pred, target)
            metrics = forward_metrics(pred, target)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metric_sums["loss"] += loss.item()
        metric_sums["mse"] += metrics["mse"]
        metric_sums["cosine"] += metrics["cosine"]
        steps += 1

    return summarize_metrics(metric_sums, steps)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()

    df, emb, merge_stats = load_airr_with_embeddings(
        airr_path=args.airr_path,
        embeddings_path=args.embeddings_path,
        locus=args.locus,
        clone_id_col=args.clone_id_col,
        embedding_column=args.embedding_column,
    )
    data_stats = validate_dataframe(df, emb, max_len=args.max_len)

    train_idx, val_idx, test_idx = split_indices(
        len(df),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_emb_raw = emb[train_idx]
    val_emb_raw = emb[val_idx]
    test_emb_raw = emb[test_idx]

    mean, std = fit_standardizer(train_emb_raw)
    train_emb = apply_standardizer(train_emb_raw, mean, std)
    val_emb = apply_standardizer(val_emb_raw, mean, std)
    test_emb = apply_standardizer(test_emb_raw, mean, std)

    train_loader = build_dataloader(
        train_df, train_emb, args.batch_size, args.max_len, True, args.num_workers
    )
    val_loader = build_dataloader(
        val_df, val_emb, args.batch_size, args.max_len, False, args.num_workers
    )
    test_loader = build_dataloader(
        test_df, test_emb, args.batch_size, args.max_len, False, args.num_workers
    )

    model = ForwardModel(output_dim=train_emb.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    save_json(
        output_dir / "data_stats.json",
        {
            **data_stats,
            **merge_stats,
            "airr_path": args.airr_path,
            "embeddings_path": args.embeddings_path,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "standardizer": {"mean_path": "mean.npy", "std_path": "std.npy"},
            "checkpoints": {"best": "best.pt", "last": "last.pt"},
        },
    )
    np.save(output_dir / "mean.npy", mean)
    np.save(output_dir / "std.npy", std)

    best_val_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            extra={"task": "forward", "max_len": args.max_len, "embedding_dim": train_emb.shape[1]},
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                extra={"task": "forward", "max_len": args.max_len, "embedding_dim": train_emb.shape[1]},
            )

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_cosine={val_metrics['cosine']:.4f}"
        )

    test_metrics = run_epoch(model, test_loader, None, device)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "test_metrics.json", test_metrics)
    print(
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_mse={test_metrics['mse']:.4f} test_cosine={test_metrics['cosine']:.4f}"
    )


if __name__ == "__main__":
    main()
