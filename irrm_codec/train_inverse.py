import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from irrm_codec.dataio import load_airr_with_embeddings
from irrm_codec.datasets import InverseDataset, collate_inverse, validate_dataframe
from irrm_codec.inverse_model import InverseModel
from irrm_codec.losses import inverse_loss, inverse_metrics
from irrm_codec.tokenization import decode
from irrm_codec.utils import (
    apply_standardizer,
    choose_device,
    fit_standardizer,
    move_to_device,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logging,
    split_indices,
    summarize_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the inverse IRRM-CODEC model.")
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--output-dir", default="artifacts/inverse")
    parser.add_argument("--locus", default="alpha")
    parser.add_argument("--clone-id-col", default="clone_id")
    parser.add_argument("--embedding-column", default="tcremp_emb")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def build_dataloader(df, emb, batch_size, max_len, shuffle, num_workers):
    dataset = InverseDataset(df, emb, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_inverse,
    )


def exact_match_rate(pred_tokens, target_tokens):
    exact_matches = 0
    total = pred_tokens.size(0)
    for pred_row, target_row in zip(pred_tokens.tolist(), target_tokens.tolist()):
        if decode(pred_row) == decode(target_row):
            exact_matches += 1
    return exact_matches / max(total, 1)


def run_epoch(model, loader, optimizer, device, stage, epoch, num_epochs, log_interval, show_progress):
    is_train = optimizer is not None
    model.train(mode=is_train)

    metric_sums = {
        "loss": 0.0,
        "token_accuracy": 0.0,
        "length_accuracy": 0.0,
        "exact_match": 0.0,
        "unk_fraction": 0.0,
    }
    steps = 0
    total_steps = len(loader)
    progress = tqdm(
        loader,
        total=total_steps,
        desc=f"{stage} {epoch}/{num_epochs}",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )

    for step, batch in enumerate(progress, start=1):
        emb, target, lengths, unk_fraction = move_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            logits, length_logits = model(emb)
            loss = inverse_loss(logits, target, length_logits, lengths)
            metrics = inverse_metrics(logits, target, length_logits, lengths)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            exact_match = 0.0
        else:
            pred_tokens, _predicted_lengths = model.generate(emb, max_len=model.max_len)
            exact_match = exact_match_rate(pred_tokens, target)

        metric_sums["loss"] += loss.item()
        metric_sums["token_accuracy"] += metrics["token_accuracy"]
        metric_sums["length_accuracy"] += metrics["length_accuracy"]
        metric_sums["exact_match"] += exact_match
        metric_sums["unk_fraction"] += float(unk_fraction.item())
        steps += 1

        should_update = step == total_steps or (log_interval > 0 and step % log_interval == 0)
        if should_update and show_progress:
            avg_metrics = summarize_metrics(metric_sums, steps)
            progress.set_postfix(
                loss=f"{avg_metrics['loss']:.4f}",
                tok_acc=f"{avg_metrics['token_accuracy']:.4f}",
                len_acc=f"{avg_metrics['length_accuracy']:.4f}",
                exact=f"{avg_metrics['exact_match']:.4f}",
            )

    if show_progress:
        progress.close()
    return summarize_metrics(metric_sums, steps)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir / "train.log")

    logger.info("starting inverse training")
    logger.info("output_dir=%s", output_dir.resolve())
    logger.info("device=%s seed=%d", device, args.seed)
    logger.info(
        "hyperparameters batch_size=%d epochs=%d lr=%.6f weight_decay=%.6f max_len=%d num_workers=%d log_interval=%d",
        args.batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.max_len,
        args.num_workers,
        args.log_interval,
    )

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

    model = InverseModel(embedding_dim=train_emb.shape[1], max_len=args.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_parameters = sum(param.numel() for param in model.parameters())
    num_trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)

    logger.info(
        "loaded data total=%d train=%d val=%d test=%d embedding_dim=%d",
        len(df),
        len(train_df),
        len(val_df),
        len(test_df),
        train_emb.shape[1],
    )
    logger.info(
        "dataloader batches train=%d val=%d test=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    logger.info(
        "model parameters total=%d trainable=%d",
        num_parameters,
        num_trainable_parameters,
    )

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
        logger.info("epoch %d/%d started", epoch, args.epochs)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            "train",
            epoch,
            args.epochs,
            args.log_interval,
            not args.no_progress,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            "val",
            epoch,
            args.epochs,
            args.log_interval,
            not args.no_progress,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            extra={"task": "inverse", "max_len": args.max_len, "embedding_dim": train_emb.shape[1]},
        )
        logger.info("saved checkpoint path=%s", output_dir / "last.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                extra={"task": "inverse", "max_len": args.max_len, "embedding_dim": train_emb.shape[1]},
            )
            logger.info("new best checkpoint path=%s val_loss=%.4f", output_dir / "best.pt", best_val_loss)

        logger.info(
            "epoch=%d summary train_loss=%.4f train_tok_acc=%.4f train_len_acc=%.4f val_loss=%.4f val_tok_acc=%.4f val_len_acc=%.4f val_exact=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["token_accuracy"],
            train_metrics["length_accuracy"],
            val_metrics["loss"],
            val_metrics["token_accuracy"],
            val_metrics["length_accuracy"],
            val_metrics["exact_match"],
        )

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    logger.info(
        "loaded best checkpoint for test path=%s epoch=%d val_loss=%.4f",
        output_dir / "best.pt",
        best_checkpoint["epoch"],
        best_checkpoint["metrics"]["loss"],
    )

    test_metrics = run_epoch(
        model,
        test_loader,
        None,
        device,
        "test",
        args.epochs,
        args.epochs,
        args.log_interval,
        not args.no_progress,
    )
    save_json(output_dir / "history.json", history)
    save_json(
        output_dir / "test_metrics.json",
        {
            **test_metrics,
            "best_checkpoint_epoch": int(best_checkpoint["epoch"]),
            "best_checkpoint_val_loss": float(best_checkpoint["metrics"]["loss"]),
        },
    )
    logger.info(
        "test summary loss=%.4f tok_acc=%.4f len_acc=%.4f exact=%.4f unk=%.4f best_epoch=%d best_val_loss=%.4f",
        test_metrics["loss"],
        test_metrics["token_accuracy"],
        test_metrics["length_accuracy"],
        test_metrics["exact_match"],
        test_metrics["unk_fraction"],
        best_checkpoint["epoch"],
        best_checkpoint["metrics"]["loss"],
    )


if __name__ == "__main__":
    main()
