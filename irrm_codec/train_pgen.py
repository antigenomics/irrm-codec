import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from irrm_codec.dataio import normalize_locus_name, read_airr_table
from irrm_codec.datasets import PgenDataset, collate_pgen, validate_target_dataframe
from irrm_codec.losses import pgen_loss, pgen_metrics
from irrm_codec.pgen_model import PgenModel
from irrm_codec.utils import (
    choose_device,
    move_to_device,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logging,
    split_indices,
    summarize_metrics,
)
from irrm_codec.wandb_utils import init_wandb_run, log_wandb_lr, log_wandb_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequence-to-log10(pgen) regression model.")
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--output-dir", default="artifacts/pgen")
    parser.add_argument("--locus")
    parser.add_argument("--clone-id-col", default="clone_id")
    parser.add_argument("--target-col", default="log10_pgen_1mm")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--mlp-hidden-dim", type=int, default=1024)
    parser.add_argument("--dilations", default="1,2,4,8")
    parser.add_argument("--encoder-type", choices=["residual", "plain_conv"], default="plain_conv")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
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


def build_dataloader(df, target_array, batch_size, max_len, shuffle, num_workers):
    dataset = PgenDataset(df, target_array, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pgen,
    )


def load_airr_with_target(airr_path, target_col, locus=None, clone_id_col="clone_id"):
    df = read_airr_table(airr_path, clone_id_col=clone_id_col)
    rows_before_locus = len(df)
    if locus is not None:
        locus = normalize_locus_name(locus)
        locus_series = df["locus"].astype(str).str.strip().str.lower().map(normalize_locus_name)
        df = df[locus_series == locus].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("AIRR table is empty after locus filtering.")
    if target_col not in df.columns:
        raise ValueError(f"AIRR table does not contain required target column {target_col!r}.")

    target_series = pd.to_numeric(df[target_col], errors="coerce")
    if target_series.isna().any():
        raise ValueError(f"Target column {target_col!r} contains NaN or non-numeric values.")

    target_array = target_series.to_numpy(dtype=np.float32)
    stats = {
        "airr_rows": int(len(df)),
        "airr_rows_before_locus": int(rows_before_locus),
        "clone_id_column": clone_id_col,
        "target_column": target_col,
        "locus": locus,
    }
    return df, target_array, stats


def build_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def build_model(args):
    dilations = tuple(int(part.strip()) for part in args.dilations.split(",") if part.strip())
    if not dilations:
        raise ValueError("dilations must contain at least one integer.")
    return PgenModel(
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.mlp_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
        dilations=dilations,
        encoder_type=args.encoder_type,
    )


def run_epoch(model, loader, optimizer, device, stage, epoch, num_epochs, log_interval, show_progress):
    is_train = optimizer is not None
    model.train(mode=is_train)

    metric_sums = {"loss": 0.0, "mse": 0.0, "rmse": 0.0, "mae": 0.0}
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
            loss = pgen_loss(pred, target)
            metrics = pgen_metrics(pred, target)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metric_sums["loss"] += loss.item()
        metric_sums["mse"] += metrics["mse"]
        metric_sums["rmse"] += metrics["rmse"]
        metric_sums["mae"] += metrics["mae"]
        steps += 1

        should_update = step == total_steps or (log_interval > 0 and step % log_interval == 0)
        if should_update and show_progress:
            avg_metrics = summarize_metrics(metric_sums, steps)
            progress.set_postfix(
                loss=f"{avg_metrics['loss']:.4f}",
                rmse=f"{avg_metrics['rmse']:.4f}",
                mae=f"{avg_metrics['mae']:.4f}",
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
    run = None

    try:
        run = init_wandb_run(
            args,
            output_dir,
            {
                "task": "pgen_regression",
                "airr_path": args.airr_path,
                "locus": args.locus,
                "clone_id_col": args.clone_id_col,
                "target_col": args.target_col,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_len": args.max_len,
                "encoder_type": args.encoder_type,
                "hidden_dim": args.hidden_dim,
                "mlp_dim": args.mlp_dim,
                "mlp_hidden_dim": args.mlp_hidden_dim,
                "dropout": args.dropout,
                "dilations": args.dilations,
                "num_workers": args.num_workers,
                "seed": args.seed,
            },
        )

        logger.info("starting pgen regression training")
        logger.info("output_dir=%s", output_dir.resolve())
        logger.info("wandb_project=%s wandb_mode=%s", args.wandb_project, args.wandb_mode)
        logger.info("device=%s seed=%d", device, args.seed)
        logger.info(
            "hyperparameters batch_size=%d epochs=%d lr=%.6f weight_decay=%.6f max_len=%d encoder_type=%s hidden_dim=%d mlp_dim=%d mlp_hidden_dim=%d dropout=%.3f dilations=%s num_workers=%d log_interval=%d early_stopping_patience=%d scheduler=%s target_col=%s",
            args.batch_size,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.max_len,
            args.encoder_type,
            args.hidden_dim,
            args.mlp_dim,
            args.mlp_hidden_dim,
            args.dropout,
            args.dilations,
            args.num_workers,
            args.log_interval,
            args.early_stopping_patience,
            args.scheduler,
            args.target_col,
        )

        df, target_array, load_stats = load_airr_with_target(
            airr_path=args.airr_path,
            target_col=args.target_col,
            locus=args.locus,
            clone_id_col=args.clone_id_col,
        )
        data_stats = validate_target_dataframe(
            df,
            target_array,
            max_len=args.max_len,
            clone_id_col=args.clone_id_col,
            target_name=args.target_col,
        )

        train_idx, val_idx, test_idx = split_indices(
            len(df),
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_target = target_array[train_idx]
        val_target = target_array[val_idx]
        test_target = target_array[test_idx]

        train_loader = build_dataloader(
            train_df, train_target, args.batch_size, args.max_len, True, args.num_workers
        )
        val_loader = build_dataloader(
            val_df, val_target, args.batch_size, args.max_len, False, args.num_workers
        )
        test_loader = build_dataloader(
            test_df, test_target, args.batch_size, args.max_len, False, args.num_workers
        )

        model = build_model(args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_scheduler(optimizer, args)
        num_parameters = sum(param.numel() for param in model.parameters())
        num_trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)

        logger.info(
            "loaded data total=%d train=%d val=%d test=%d",
            len(df),
            len(train_df),
            len(val_df),
            len(test_df),
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
                **load_stats,
                "airr_path": args.airr_path,
                "seed": int(args.seed),
                "train_fraction": float(args.train_fraction),
                "val_fraction": float(args.val_fraction),
                "train_size": int(len(train_df)),
                "val_size": int(len(val_df)),
                "test_size": int(len(test_df)),
                "checkpoints": {"best": "best.pt", "last": "last.pt"},
            },
        )

        checkpoint_extra = {
            "task": "pgen_regression",
            "max_len": args.max_len,
            "target_col": args.target_col,
            "encoder_type": args.encoder_type,
            "hidden_dim": args.hidden_dim,
            "mlp_dim": args.mlp_dim,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "dropout": args.dropout,
            "dilations": args.dilations,
        }

        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
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

            if scheduler is not None:
                scheduler.step(val_metrics["loss"])

            logger.info(
                "saved checkpoint path=%s", output_dir / "last.pt"
            )
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                extra=checkpoint_extra,
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    val_metrics,
                    extra=checkpoint_extra,
                )
                logger.info("new best checkpoint path=%s val_loss=%.4f", output_dir / "best.pt", best_val_loss)
            else:
                epochs_without_improvement += 1

            logger.info(
                "epoch=%d summary train_loss=%.4f train_rmse=%.4f train_mae=%.4f val_loss=%.4f val_rmse=%.4f val_mae=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["rmse"],
                train_metrics["mae"],
                val_metrics["loss"],
                val_metrics["rmse"],
                val_metrics["mae"],
            )
            logger.info(
                "epoch=%d control best_epoch=%d best_val_loss=%.4f epochs_without_improvement=%d lr=%.6g",
                epoch,
                best_epoch,
                best_val_loss,
                epochs_without_improvement,
                optimizer.param_groups[0]["lr"],
            )

            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                logger.info(
                    "early stopping triggered at epoch=%d best_epoch=%d best_val_loss=%.4f patience=%d",
                    epoch,
                    best_epoch,
                    best_val_loss,
                    args.early_stopping_patience,
                )
                break

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
        log_wandb_metrics(run, "test", test_metrics, len(history))
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
            "test summary loss=%.4f rmse=%.4f mae=%.4f best_epoch=%d best_val_loss=%.4f",
            test_metrics["loss"],
            test_metrics["rmse"],
            test_metrics["mae"],
            best_checkpoint["epoch"],
            best_checkpoint["metrics"]["loss"],
        )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
