import argparse
import logging
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
    choose_device,
    move_to_device,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logging,
    summarize_metrics,
)
from irrm_codec.wandb_utils import init_wandb_run, log_wandb_lr, log_wandb_metrics


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
    parser.add_argument("--wandb-project", default="irrm-codec")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-dir", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


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
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    logger = logging.getLogger("irrm_codec")
    logger.info("run_epoch start stage=%s epoch=%d/%d batches=%d", stage, epoch, num_epochs, len(loader))

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
        emb, decoder_input, target, lengths, unk_fraction = move_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            logits, length_logits = model(emb, decoder_input)
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
    logger.info("run_epoch done stage=%s epoch=%d/%d", stage, epoch, num_epochs)
    return summarize_metrics(metric_sums, steps)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir / "train.log")
    run = None

    try:
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
        run = init_wandb_run(
            args,
            output_dir,
            {
                "task": "inverse",
                "airr_path": args.airr_path,
                "embeddings_path": args.embeddings_path,
                "locus": args.locus,
                "clone_id_col": args.clone_id_col,
                "embedding_column": args.embedding_column,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_len": args.max_len,
                "num_workers": args.num_workers,
                "seed": args.seed,
            },
        )
        logger.info("wandb_project=%s wandb_mode=%s", args.wandb_project, args.wandb_mode)

        df, emb_array, merge_stats = load_airr_with_embeddings(
            airr_path=args.airr_path,
            embeddings_path=args.embeddings_path,
            locus=args.locus,
            clone_id_col=args.clone_id_col,
            embedding_column=args.embedding_column,
        )
        data_stats = validate_dataframe(df, emb_array, max_len=args.max_len, clone_id_col=args.clone_id_col)

        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(len(df))
        train_end = int(len(df) * args.train_fraction)
        val_end = train_end + int(len(df) * args.val_fraction)
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_emb = emb_array[train_idx]
        val_emb = emb_array[val_idx]
        test_emb = emb_array[test_idx]
        split_row_counts = {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        }

        mean = train_emb.mean(axis=0).astype(np.float32)
        std = train_emb.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

        train_loader = DataLoader(
            InverseDataset(train_df, (train_emb - mean) / std, max_len=args.max_len),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_inverse,
        )
        val_loader = DataLoader(
            InverseDataset(val_df, (val_emb - mean) / std, max_len=args.max_len),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_inverse,
        )
        test_loader = DataLoader(
            InverseDataset(test_df, (test_emb - mean) / std, max_len=args.max_len),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_inverse,
        )

        model = InverseModel(embedding_dim=merge_stats["embedding_dim"], max_len=args.max_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        num_parameters = sum(param.numel() for param in model.parameters())
        num_trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)

        logger.info(
            "loaded data total=%d train=%d val=%d test=%d embedding_dim=%d",
            data_stats["num_samples"],
            split_row_counts["train"],
            split_row_counts["val"],
            split_row_counts["test"],
            merge_stats["embedding_dim"],
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
                "train_size": int(split_row_counts["train"]),
                "val_size": int(split_row_counts["val"]),
                "test_size": int(split_row_counts["test"]),
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
            log_wandb_metrics(run, "train", train_metrics, epoch)
            log_wandb_metrics(run, "val", val_metrics, epoch)
            log_wandb_lr(run, optimizer, epoch)

            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                extra={"task": "inverse", "max_len": args.max_len, "embedding_dim": merge_stats["embedding_dim"]},
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
                    extra={"task": "inverse", "max_len": args.max_len, "embedding_dim": merge_stats["embedding_dim"]},
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
        log_wandb_metrics(run, "test", test_metrics, len(history))
        save_json(output_dir / "history.json", history)
        save_json(output_dir / "test_metrics.json", test_metrics)
        logger.info(
            "test summary loss=%.4f tok_acc=%.4f len_acc=%.4f exact=%.4f unk=%.4f",
            test_metrics["loss"],
            test_metrics["token_accuracy"],
            test_metrics["length_accuracy"],
            test_metrics["exact_match"],
            test_metrics["unk_fraction"],
        )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
