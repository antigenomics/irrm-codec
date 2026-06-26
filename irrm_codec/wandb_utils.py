from pathlib import Path

import wandb


def init_wandb_run(args, output_dir, config):
    run_name = args.wandb_run_name or Path(output_dir).name
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        dir=str(Path(args.wandb_dir) if args.wandb_dir else output_dir),
        mode=args.wandb_mode,
        config=config,
    )
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("test/*", step_metric="epoch")
    return run


def log_wandb_metrics(run, stage, metrics, epoch):
    payload = {"epoch": int(epoch)}
    payload.update({f"{stage}/{name}": float(value) for name, value in metrics.items()})
    run.log(payload)


def log_wandb_lr(run, optimizer, epoch):
    if optimizer is None:
        return
    run.log({"epoch": int(epoch), "train/lr": float(optimizer.param_groups[0]["lr"])})
