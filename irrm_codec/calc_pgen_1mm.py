import argparse
import json
import logging
import math
import inspect
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from irrm_codec.dataio import normalize_locus_name, read_airr_table


_thread_state = threading.local()


def _setup_logging(log_path=None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("irrm_codec.calc_pgen_1mm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute 1-mismatch pgen values for AIRR clonotypes using mirpy OLGA models."
    )
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-stats-path")
    parser.add_argument("--locus")
    parser.add_argument("--chain", default="TRB")
    parser.add_argument("--species", default="human")
    parser.add_argument("--model-path")
    parser.add_argument("--clone-id-col", default="clone_id")
    parser.add_argument("--cdr3-col", default="junction_aa")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--pgen-col", default="pgen_1mm")
    parser.add_argument("--log10-pgen-col", default="log10_pgen_1mm")
    parser.add_argument(
        "--is-d-present",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether the OLGA model contains a D segment. Default: infer from chain.",
    )
    parser.add_argument(
        "--mirpy-path",
        default=None,
        help="Optional path to a local mirpy checkout. Defaults to ../mirpy when present.",
    )
    parser.add_argument("--log-path")
    return parser.parse_args()


def _resolve_default_mirpy_path():
    candidate = Path(__file__).resolve().parents[1].parent / "mirpy"
    return candidate if candidate.exists() else None


def _import_olga_model_class(mirpy_path=None):
    try:
        from mir.basic.pgen import OlgaModel

        return OlgaModel
    except ModuleNotFoundError:
        resolved = Path(mirpy_path).resolve() if mirpy_path else _resolve_default_mirpy_path()
        if resolved is not None and resolved.exists():
            resolved_str = str(resolved)
            if resolved_str not in sys.path:
                sys.path.insert(0, resolved_str)
            from mir.basic.pgen import OlgaModel

            return OlgaModel
        raise ModuleNotFoundError(
            "Failed to import mir.basic.pgen.OlgaModel. "
            "Install mirpy-lib in the environment or pass --mirpy-path to a local checkout."
        )


def _parse_is_d_present(value):
    if value == "auto":
        return None
    return value == "true"


def _chunk_bounds(num_items, num_chunks):
    if num_items == 0:
        return []
    num_chunks = max(1, min(num_chunks, num_items))
    bounds = []
    for chunk_idx in range(num_chunks):
        start = (chunk_idx * num_items) // num_chunks
        end = ((chunk_idx + 1) * num_items) // num_chunks
        if start < end:
            bounds.append((start, end))
    return bounds


def _build_olga_model(OlgaModel, chain, species, model_path, is_d_present):
    signature = inspect.signature(OlgaModel.__init__)
    parameter_names = set(signature.parameters)

    if "locus" in parameter_names:
        kwargs = {"locus": chain.upper(), "species": species.lower()}
        if model_path:
            kwargs["model"] = model_path
        if is_d_present is not None:
            kwargs["is_d_present"] = is_d_present
        return OlgaModel(**kwargs)

    kwargs = {"chain": chain.upper()}
    if model_path:
        kwargs["model"] = model_path
    if is_d_present is not None:
        kwargs["is_d_present"] = is_d_present
    return OlgaModel(**kwargs)


def _compute_pgen_1mm(model, seq):
    if hasattr(model, "compute_pgen_junction_aa_1mm"):
        return model.compute_pgen_junction_aa_1mm(seq)
    if hasattr(model, "compute_pgen_cdr3aa_1mm"):
        return model.compute_pgen_cdr3aa_1mm(seq)
    raise AttributeError(
        "OlgaModel does not provide a supported 1mm pgen method. "
        "Expected compute_pgen_junction_aa_1mm or compute_pgen_cdr3aa_1mm."
    )


def _get_thread_model(chain, species, model_path, is_d_present, mirpy_path):
    cache_key = (chain.upper(), species.lower(), model_path, is_d_present, mirpy_path)
    model_cache = getattr(_thread_state, "model_cache", None)
    if model_cache is None:
        model_cache = {}
        _thread_state.model_cache = model_cache
    if cache_key not in model_cache:
        OlgaModel = _import_olga_model_class(mirpy_path)
        model_cache[cache_key] = _build_olga_model(
            OlgaModel=OlgaModel,
            chain=chain,
            species=species,
            model_path=model_path,
            is_d_present=is_d_present,
        )
    return model_cache[cache_key]


def _compute_chunk(
    chunk_id,
    start,
    end,
    sequences,
    chain,
    species,
    model_path,
    is_d_present,
    mirpy_path,
    batch_size,
):
    model = _get_thread_model(chain, species, model_path, is_d_present, mirpy_path)
    values = np.empty(end - start, dtype=np.float64)
    offset = 0
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        for seq in sequences[batch_start:batch_end]:
            values[offset] = _compute_pgen_1mm(model, seq)
            offset += 1
    return chunk_id, start, values


def _read_and_filter_airr(args):
    df = read_airr_table(args.airr_path, clone_id_col=args.clone_id_col)
    rows_before_locus = len(df)
    if args.locus is not None:
        locus = normalize_locus_name(args.locus)
        locus_series = df["locus"].astype(str).str.strip().str.lower().map(normalize_locus_name)
        df = df.loc[locus_series == locus].copy()
    else:
        locus = None

    if len(df) == 0:
        raise ValueError("AIRR table is empty after locus filtering.")
    if args.cdr3_col not in df.columns:
        raise ValueError(f"AIRR table does not contain required cdr3 column {args.cdr3_col!r}.")

    raw_cdr3 = df[args.cdr3_col]
    if raw_cdr3.isna().any():
        missing = int(raw_cdr3.isna().sum())
        raise ValueError(f"AIRR table contains {missing} missing values in {args.cdr3_col!r}.")

    sequences = raw_cdr3.astype(str).str.strip().str.upper()
    invalid_mask = sequences.eq("")
    if invalid_mask.any():
        raise ValueError(
            f"AIRR table contains {int(invalid_mask.sum())} empty CDR3 values in {args.cdr3_col!r}."
        )

    return df.reset_index(drop=True), sequences.tolist(), {
        "airr_path": str(Path(args.airr_path).resolve()),
        "rows_before_locus_filter": int(rows_before_locus),
        "rows_after_locus_filter": int(len(df)),
        "locus": locus,
        "chain": args.chain.upper(),
        "species": args.species.lower(),
        "cdr3_column": args.cdr3_col,
        "clone_id_column": args.clone_id_col,
    }


def _log10_or_neg_inf(value):
    if value > 0:
        return math.log10(value)
    return float("-inf")


def _save_table(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix in {".tsv", ".airr"}:
        df.to_csv(output_path, sep="\t", index=False)
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(
            f"Unsupported output extension {suffix!r}. Use .parquet, .tsv, .airr or .csv."
        )


def _save_stats(stats_path, payload):
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    output_path = Path(args.output_path)
    default_log_path = output_path.with_suffix(output_path.suffix + ".log")
    logger = _setup_logging(args.log_path or default_log_path)
    logger.info("starting 1mm pgen calculation")

    df, sequences, input_stats = _read_and_filter_airr(args)
    num_rows = len(df)
    chunk_bounds = _chunk_bounds(num_rows, args.threads)
    pgen_values = np.empty(num_rows, dtype=np.float64)
    is_d_present = _parse_is_d_present(args.is_d_present)

    logger.info(
        "loaded rows=%d chain=%s locus=%s threads=%d batch_size=%d",
        num_rows,
        args.chain.upper(),
        input_stats["locus"],
        len(chunk_bounds),
        args.batch_size,
    )

    with ThreadPoolExecutor(max_workers=len(chunk_bounds) or 1) as executor:
        futures = [
            executor.submit(
                _compute_chunk,
                chunk_id,
                start,
                end,
                sequences,
                args.chain,
                args.species,
                args.model_path,
                is_d_present,
                args.mirpy_path,
                args.batch_size,
            )
            for chunk_id, (start, end) in enumerate(chunk_bounds)
        ]
        for future in futures:
            chunk_id, start, values = future.result()
            pgen_values[start : start + len(values)] = values
            logger.info(
                "completed chunk=%d start=%d end=%d rows=%d",
                chunk_id,
                start,
                start + len(values),
                len(values),
            )

    df[args.pgen_col] = pgen_values
    df[args.log10_pgen_col] = np.array([_log10_or_neg_inf(value) for value in pgen_values], dtype=np.float64)
    _save_table(df, output_path)

    stats_payload = {
        **input_stats,
        "output_path": str(output_path.resolve()),
        "stats_path": str(
            Path(args.output_stats_path).resolve()
            if args.output_stats_path
            else output_path.with_suffix(output_path.suffix + ".stats.json").resolve()
        ),
        "rows_written": int(len(df)),
        "threads": int(len(chunk_bounds)),
        "batch_size": int(args.batch_size),
        "model_path": str(Path(args.model_path).resolve()) if args.model_path else None,
        "mirpy_path": str(Path(args.mirpy_path).resolve()) if args.mirpy_path else None,
        "is_d_present": is_d_present,
        "pgen_column": args.pgen_col,
        "log10_pgen_column": args.log10_pgen_col,
    }
    stats_path = args.output_stats_path or output_path.with_suffix(output_path.suffix + ".stats.json")
    _save_stats(stats_path, stats_payload)
    logger.info("saved output=%s", output_path.resolve())
    logger.info("saved stats=%s", Path(stats_path).resolve())


if __name__ == "__main__":
    main()
