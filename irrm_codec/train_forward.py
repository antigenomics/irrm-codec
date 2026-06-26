import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from irrm_codec.batch_cache import cleanup_batch_cache, prepare_cached_training_data, save_training_metadata
from irrm_codec.datasets import collate_forward
from irrm_codec.forward_model import ForwardModel
from irrm_codec.losses import forward_loss, forward_metrics
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
    parser.add_argument("--reader-batch-size", type=int, default=4096)
    parser.add_argument("--cache-batch-size", type=int, default=4096)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--wandb-project", default="irrm-codec")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-dir", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, optimizer, device, stage, epoch, num_epochs, log_interval, show_progress):
    is_train = optimizer is not None
    model.train(mode=is_train)
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    logger = logging.getLogger("irrm_codec")
    logger.info("run_epoch start stage=%s epoch=%d/%d batches=%d", stage, epoch, num_epochs, len(loader))

    metric_sums = {"loss": 0.0, "mse": 0.0, "cosine": 0.0}
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

        should_update = step == total_steps or (log_interval > 0 and step % log_interval == 0)
        if should_update and show_progress:
            avg_metrics = summarize_metrics(metric_sums, steps)
            progress.set_postfix(
                loss=f"{avg_metrics['loss']:.4f}",
                mse=f"{avg_metrics['mse']:.4f}",
                cosine=f"{avg_metrics['cosine']:.4f}",
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
    cache_dir = None
    run = None

    try:
        logger.info("starting forward training")
        logger.info("output_dir=%s", output_dir.resolve())
        logger.info("device=%s seed=%d", device, args.seed)
        logger.info(
            "hyperparameters batch_size=%d reader_batch_size=%d cache_batch_size=%d epochs=%d lr=%.6f weight_decay=%.6f max_len=%d num_workers=%d log_interval=%d",
            args.batch_size,
            args.reader_batch_size,
            args.cache_batch_size,
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
                "task": "forward",
                "airr_path": args.airr_path,
                "embeddings_path": args.embeddings_path,
                "locus": args.locus,
                "clone_id_col": args.clone_id_col,
                "embedding_column": args.embedding_column,
                "batch_size": args.batch_size,
                "reader_batch_size": args.reader_batch_size,
                "cache_batch_size": args.cache_batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_len": args.max_len,
                "num_workers": args.num_workers,
                "seed": args.seed,
            },
        )
        logger.info("wandb_project=%s wandb_mode=%s", args.wandb_project, args.wandb_mode)

        prepared = prepare_cached_training_data(
            args,
            logger,
            task="forward",
            collate_fn=collate_forward,
        )
        manifest = prepared["manifest"]
        mean = prepared["mean"]
        std = prepared["std"]
        data_stats = prepared["data_stats"]
        merge_stats = prepared["merge_stats"]
        split_row_counts = prepared["split_row_counts"]
        cache_dir = prepared["cache_dir"]
        train_loader = prepared["train_loader"]
        val_loader = prepared["val_loader"]
        test_loader = prepared["test_loader"]

        model = ForwardModel(output_dim=merge_stats["embedding_dim"], max_len=args.max_len).to(device)
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

        save_training_metadata(
            output_dir,
            args,
            data_stats,
            merge_stats,
            split_row_counts,
            manifest,
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
                extra={"task": "forward", "max_len": args.max_len, "embedding_dim": merge_stats["embedding_dim"]},
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
                    extra={"task": "forward", "max_len": args.max_len, "embedding_dim": merge_stats["embedding_dim"]},
                )
                logger.info("new best checkpoint path=%s val_loss=%.4f", output_dir / "best.pt", best_val_loss)

            logger.info(
                "epoch=%d summary train_loss=%.4f train_mse=%.4f train_cosine=%.4f val_loss=%.4f val_mse=%.4f val_cosine=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["mse"],
                train_metrics["cosine"],
                val_metrics["loss"],
                val_metrics["mse"],
                val_metrics["cosine"],
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
            "test summary loss=%.4f mse=%.4f cosine=%.4f",
            test_metrics["loss"],
            test_metrics["mse"],
            test_metrics["cosine"],
        )
    finally:
        if run is not None:
            run.finish()
        if cache_dir is not None:
            cleanup_batch_cache(cache_dir, logger=logger)


if __name__ == "__main__":
    main()
