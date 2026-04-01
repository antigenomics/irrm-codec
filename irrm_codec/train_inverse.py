import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from irrm_codec.datasets import collate_inverse, validate_airr_dataframe
from irrm_codec.inverse_model import InverseModel
from irrm_codec.losses import inverse_loss, inverse_metrics
from irrm_codec.streaming_training import (
    build_streaming_dataloader,
    fit_streaming_standardizer_with_logging,
    prepare_streaming_splits,
)
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
    parser.add_argument("--reader-batch-size", type=int, default=4096)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def build_dataloader(
    records_by_key,
    selected_keys,
    iter_batches_fn,
    batch_size,
    max_len,
    shuffle,
    num_workers,
    mean,
    std,
    seed,
):
    return build_streaming_dataloader(
        task="inverse",
        collate_fn=collate_inverse,
        records_by_key=records_by_key,
        selected_keys=selected_keys,
        iter_embedding_batches_fn=iter_batches_fn,
        batch_size=batch_size,
        max_len=max_len,
        shuffle=shuffle,
        num_workers=num_workers,
        mean=mean,
        std=std,
        seed=seed,
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

    logger.info("starting inverse training")
    logger.info("output_dir=%s", output_dir.resolve())
    logger.info("device=%s seed=%d", device, args.seed)
    logger.info(
        "hyperparameters batch_size=%d reader_batch_size=%d epochs=%d lr=%.6f weight_decay=%.6f max_len=%d num_workers=%d log_interval=%d",
        args.batch_size,
        args.reader_batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.max_len,
        args.num_workers,
        args.log_interval,
    )

    logger.info("preparing streaming splits")
    df, records_by_key, split_keys, merge_stats, iter_batches_fn = prepare_streaming_splits(args, logger=logger)
    data_stats = validate_airr_dataframe(df, max_len=args.max_len)
    data_stats["embedding_dim"] = merge_stats["embedding_dim"]

    mean, std = fit_streaming_standardizer_with_logging(
        iter_batches_fn,
        split_keys["train"],
        merge_stats["embedding_dim"],
        logger=logger,
    )

    logger.info("building dataloaders")
    logger.info("creating train dataloader")
    train_loader = build_dataloader(
        records_by_key,
        split_keys["train"],
        iter_batches_fn,
        args.batch_size,
        args.max_len,
        True,
        args.num_workers,
        mean,
        std,
        args.seed,
    )
    logger.info("creating val dataloader")
    val_loader = build_dataloader(
        records_by_key,
        split_keys["val"],
        iter_batches_fn,
        args.batch_size,
        args.max_len,
        False,
        args.num_workers,
        mean,
        std,
        args.seed,
    )
    logger.info("creating test dataloader")
    test_loader = build_dataloader(
        records_by_key,
        split_keys["test"],
        iter_batches_fn,
        args.batch_size,
        args.max_len,
        False,
        args.num_workers,
        mean,
        std,
        args.seed,
    )
    logger.info("dataloaders ready; starting model initialization")

    model = InverseModel(embedding_dim=merge_stats["embedding_dim"], max_len=args.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_parameters = sum(param.numel() for param in model.parameters())
    num_trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)

    logger.info(
        "loaded data total=%d train=%d val=%d test=%d embedding_dim=%d",
        len(df),
        len(split_keys["train"]),
        len(split_keys["val"]),
        len(split_keys["test"]),
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
            "train_size": int(len(split_keys["train"])),
            "val_size": int(len(split_keys["val"])),
            "test_size": int(len(split_keys["test"])),
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


if __name__ == "__main__":
    main()
