import argparse
import itertools
import subprocess
from pathlib import Path


FORWARD_GRID = {
    "token_embedding_dim": [64],
    "hidden_dim": [192, 256],
    "mlp_dim": [512, 768],
    "mlp_hidden_dim": [1024],
    "dropout": [0.1, 0.2],
    "dilations": ["1,2,4,8", "1,2,4,8,16"],
    "encoder_type": ["residual", "plain_conv"],
}

INVERSE_GRID = {
    "hidden_dim": [384, 512, 768],
    "dropout": [0.1, 0.2],
    "num_layers": [2, 3, 4],
    "nhead": [8],
    "ff_mult": [4],
}

CHAINS = ("TRB", "TRA", "TRG", "TRD", "IGH", "IGK", "IGL")


def parse_args():
    parser = argparse.ArgumentParser(description="Submit W&B hyperparameter grid jobs to Slurm.")
    parser.add_argument("--model", choices=["forward", "inverse", "both"], default="both")
    parser.add_argument("--chains", nargs="+", default=list(CHAINS))
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--airr-dir", default="/projects/immunestatus/vdjrearm/airr_format")
    parser.add_argument("--embeddings-dir", default="/projects/immunestatus/vdjrearm/tcremp")
    parser.add_argument("--output-root", default="/projects/immunestatus/vdjrearm/irrmcodec/wandb_sweeps")
    parser.add_argument("--epochs-forward", type=int, default=40)
    parser.add_argument("--epochs-inverse", type=int, default=40)
    parser.add_argument("--batch-size-forward", type=int, default=256)
    parser.add_argument("--batch-size-inverse", type=int, default=128)
    parser.add_argument("--lr-forward", type=float, default=1e-3)
    parser.add_argument("--lr-inverse", type=float, default=3e-4)
    parser.add_argument("--weight-decay-forward", type=float, default=1e-4)
    parser.add_argument("--weight-decay-inverse", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-model limit on submitted configs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iter_grid(grid):
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def slugify_params(params):
    parts = []
    for key, value in params.items():
        short_key = (
            key.replace("token_embedding_dim", "ted")
            .replace("hidden_dim", "hd")
            .replace("mlp_hidden_dim", "mhd")
            .replace("mlp_dim", "md")
            .replace("dropout", "do")
            .replace("dilations", "dil")
            .replace("encoder_type", "enc")
            .replace("num_layers", "layers")
            .replace("nhead", "head")
            .replace("ff_mult", "ff")
        )
        value_str = str(value).replace(",", "-").replace(".", "p")
        parts.append(f"{short_key}{value_str}")
    slug = "_".join(parts)
    return slug[:180]


def build_export(args, model, chain, params):
    export = {
        "ALL": None,
        "REPO_ROOT": str(Path(__file__).resolve().parents[1]),
        "CHAIN": chain,
        "PYTHON_BIN": args.python_bin,
        "AIRR_DIR": args.airr_dir,
        "EMBEDDINGS_DIR": args.embeddings_dir,
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_MODE": args.wandb_mode,
        "SEED": str(args.seed),
        "NUM_WORKERS": str(args.num_workers),
        "LOG_INTERVAL": str(args.log_interval),
        "RUN_SLUG": slugify_params(params),
    }

    if model == "forward":
        export["OUTPUT_ROOT"] = f"{args.output_root}/forward"
        export["WANDB_PROJECT"] = f"irrm-codec-forward-{chain.lower()}"
        export["EPOCHS"] = str(args.epochs_forward)
        export["BATCH_SIZE"] = str(args.batch_size_forward)
        export["LR"] = str(args.lr_forward)
        export["WEIGHT_DECAY"] = str(args.weight_decay_forward)
        export["TOKEN_EMBEDDING_DIM"] = str(params["token_embedding_dim"])
        export["HIDDEN_DIM"] = str(params["hidden_dim"])
        export["MLP_DIM"] = str(params["mlp_dim"])
        export["MLP_HIDDEN_DIM"] = str(params["mlp_hidden_dim"])
        export["DROPOUT"] = str(params["dropout"])
        export["DILATIONS"] = str(params["dilations"]).replace(",", ":")
        export["ENCODER_TYPE"] = str(params["encoder_type"])
    else:
        export["OUTPUT_ROOT"] = f"{args.output_root}/inverse"
        export["WANDB_PROJECT"] = f"irrm-codec-inverse-{chain.lower()}"
        export["EPOCHS"] = str(args.epochs_inverse)
        export["BATCH_SIZE"] = str(args.batch_size_inverse)
        export["LR"] = str(args.lr_inverse)
        export["WEIGHT_DECAY"] = str(args.weight_decay_inverse)
        export["HIDDEN_DIM"] = str(params["hidden_dim"])
        export["DROPOUT"] = str(params["dropout"])
        export["NUM_LAYERS"] = str(params["num_layers"])
        export["NHEAD"] = str(params["nhead"])
        export["FF_MULT"] = str(params["ff_mult"])
    return export


def export_to_arg(export):
    parts = []
    for key, value in export.items():
        if value is None:
            parts.append(key)
        else:
            parts.append(f"{key}={value}")
    return ",".join(parts)


def submit_model(args, model, grid, sbatch_path):
    submitted = 0
    for index, params in enumerate(iter_grid(grid), start=1):
        if args.limit and index > args.limit:
            break
        for chain in args.chains:
            export = build_export(args, model, chain.upper(), params)
            command = ["sbatch", f"--export={export_to_arg(export)}", str(sbatch_path)]
            if args.dry_run:
                print(" ".join(command))
            else:
                subprocess.run(command, check=True)
            submitted += 1
    return submitted


def main():
    args = parse_args()
    scripts_root = Path(__file__).resolve().parents[1]
    total = 0

    if args.model in {"forward", "both"}:
        total += submit_model(
            args,
            "forward",
            FORWARD_GRID,
            scripts_root / "slurm" / "train_forward_wandb_grid.sbatch",
        )
    if args.model in {"inverse", "both"}:
        total += submit_model(
            args,
            "inverse",
            INVERSE_GRID,
            scripts_root / "slurm" / "train_inverse_wandb_grid.sbatch",
        )

    print(f"submitted_jobs={total}")


if __name__ == "__main__":
    main()
