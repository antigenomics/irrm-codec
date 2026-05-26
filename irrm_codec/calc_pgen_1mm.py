import argparse
import inspect
import json
import logging
import math
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from irrm_codec.dataio import normalize_locus_name, read_airr_table


_MODEL_CACHE = {}


def parse_args():
    p = argparse.ArgumentParser(description="Compute exact and 1mm pgen values for an AIRR table.")
    p.add_argument("--airr-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--output-stats-path")
    p.add_argument("--locus")
    p.add_argument("--chain", default="TRB")
    p.add_argument("--species", default="human")
    p.add_argument("--model-path")
    p.add_argument("--clone-id-col", default="clone_id")
    p.add_argument("--cdr3-col", default="junction_aa")
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024, help=argparse.SUPPRESS)
    p.add_argument("--exact-pgen-col", default="pgen")
    p.add_argument("--pgen-col", default="pgen_1mm")
    p.add_argument("--exact-log10-pgen-col", default="log10_pgen")
    p.add_argument("--log10-pgen-col", default="log10_pgen_1mm")
    p.add_argument("--is-d-present", choices=["auto", "true", "false"], default="auto")
    p.add_argument("--mirpy-path")
    p.add_argument("--log-path")
    return p.parse_args()


def _logger(log_path):
    handlers = [logging.StreamHandler()]
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("irrm_codec.calc_pgen_1mm")


def _load_airr(path, clone_id_col, cdr3_col, locus):
    df = read_airr_table(path, clone_id_col=clone_id_col)
    before = len(df)
    if locus is not None:
        locus = normalize_locus_name(locus)
        mask = df["locus"].astype(str).str.strip().str.lower().map(normalize_locus_name).eq(locus)
        df = df.loc[mask].copy()
    if df.empty:
        raise ValueError("AIRR table is empty after locus filtering.")
    if cdr3_col not in df.columns:
        raise ValueError(f"AIRR table does not contain required cdr3 column {cdr3_col!r}.")
    seq = df[cdr3_col]
    if seq.isna().any():
        raise ValueError(f"AIRR table contains missing values in {cdr3_col!r}.")
    seq = seq.astype(str).str.strip().str.upper()
    if seq.eq("").any():
        raise ValueError(f"AIRR table contains empty values in {cdr3_col!r}.")
    return df.reset_index(drop=True), seq.tolist(), {
        "airr_path": str(Path(path).resolve()),
        "rows_before_locus_filter": int(before),
        "rows_after_locus_filter": int(len(df)),
        "locus": locus,
        "cdr3_column": cdr3_col,
        "clone_id_column": clone_id_col,
    }


def _chunk_bounds(n, size):
    if size < 1:
        raise ValueError("--chunk-size must be >= 1.")
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def _chunk_dir(output_path):
    output_path = Path(output_path)
    return output_path.parent / f"{output_path.name}.chunks"


def _chunk_path(output_path, chunk_id):
    return _chunk_dir(output_path) / f"chunk_{chunk_id:04d}.tsv"


def _save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _log10(x):
    return math.log10(x) if x > 0 else float("-inf")


def _import_olga(cfg):
    log = logging.getLogger("irrm_codec.calc_pgen_1mm")
    log.info("importing OlgaModel mirpy_path=%s", cfg["mirpy_path"] or "<default>")
    try:
        from mir.basic.pgen import OlgaModel
        log.info("imported OlgaModel from installed package")
        return OlgaModel
    except ModuleNotFoundError:
        if cfg["mirpy_path"]:
            mirpy = Path(cfg["mirpy_path"]).resolve()
        else:
            mirpy = Path(__file__).resolve().parents[2] / "mirpy"
        log.info("installed package missing, trying local mirpy path=%s", mirpy)
        if mirpy.exists():
            sys.path.insert(0, str(mirpy))
            from mir.basic.pgen import OlgaModel
            log.info("imported OlgaModel from local mirpy path=%s", mirpy)
            return OlgaModel
        raise


def _model(cfg):
    key = tuple(cfg[k] for k in ("chain", "species", "model_path", "is_d_present", "mirpy_path"))
    if key not in _MODEL_CACHE:
        log = logging.getLogger("irrm_codec.calc_pgen_1mm")
        log.info(
            "building OlgaModel chain=%s species=%s model_path=%s is_d_present=%s",
            cfg["chain"],
            cfg["species"],
            cfg["model_path"] or "<default>",
            cfg["is_d_present"],
        )
        OlgaModel = _import_olga(cfg)
        sig = set(inspect.signature(OlgaModel.__init__).parameters)
        kwargs = {"species": cfg["species"].lower(), "model": cfg["model_path"]}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if "locus" in sig:
            kwargs["locus"] = cfg["chain"].upper()
        else:
            kwargs["chain"] = cfg["chain"].upper()
        if cfg["is_d_present"] is not None:
            kwargs["is_d_present"] = cfg["is_d_present"]
        _MODEL_CACHE[key] = OlgaModel(**kwargs)
        log.info("OlgaModel ready chain=%s", cfg["chain"])
    return _MODEL_CACHE[key]


def _compute(model, seq):
    exact_fn = getattr(model, "compute_pgen_junction_aa", None) or getattr(model, "compute_pgen_cdr3aa")
    mm_fn = getattr(model, "compute_pgen_junction_aa_1mm", None) or getattr(model, "compute_pgen_cdr3aa_1mm")
    return exact_fn(seq), mm_fn(seq)


def _read_chunk(output_path, chunk_id, seqs, cols):
    path = _chunk_path(output_path, chunk_id)
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    if list(df.columns) != cols or len(df) != len(seqs):
        raise ValueError(f"Unexpected chunk contents in {path}.")
    if not np.array_equal(df[cols[0]].astype(str).to_numpy(dtype=object), np.asarray(seqs, dtype=object)):
        raise ValueError(f"Chunk sequences do not match expected values in {path}.")
    return df[cols[1]].to_numpy(dtype=np.float64), df[cols[2]].to_numpy(dtype=np.float64)


def _write_chunk(output_path, chunk_id, seqs, exact, mm, cols):
    _chunk_dir(output_path).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            cols[0]: seqs,
            cols[1]: np.array([_log10(x) for x in exact], dtype=np.float64),
            cols[2]: np.array([_log10(x) for x in mm], dtype=np.float64),
        }
    ).to_csv(_chunk_path(output_path, chunk_id), sep="\t", index=False)


def _worker(worker_id, jobs, cfg):
    _logger(cfg["log_path"])
    log = logging.getLogger("irrm_codec.calc_pgen_1mm")
    log.info("worker=%d started jobs=%d", worker_id, len(jobs))
    log.info("worker=%d loading AIRR from %s", worker_id, cfg["airr_path"])
    _, sequences, _ = _load_airr(cfg["airr_path"], cfg["clone_id_col"], cfg["cdr3_col"], cfg["locus"])
    log.info("worker=%d loaded sequences=%d", worker_id, len(sequences))
    model = _model(cfg)
    cols = [cfg["cdr3_col"], cfg["exact_log10_pgen_col"], cfg["log10_pgen_col"]]
    done = []
    for chunk_id, start, end in jobs:
        seqs = sequences[start:end]
        log.info("worker=%d entering chunk=%d start=%d end=%d", worker_id, chunk_id, start, end)
        if _read_chunk(cfg["output_path"], chunk_id, seqs, cols) is None:
            log.info("worker=%d computing chunk=%d rows=%d", worker_id, chunk_id, len(seqs))
            exact = np.empty(len(seqs), dtype=np.float64)
            mm = np.empty(len(seqs), dtype=np.float64)
            for i, seq in enumerate(seqs):
                if i == 0:
                    log.info("worker=%d computing first sequence for chunk=%d seq_len=%d", worker_id, chunk_id, len(seq))
                exact[i], mm[i] = _compute(model, seq)
            log.info("worker=%d writing chunk=%d", worker_id, chunk_id)
            _write_chunk(cfg["output_path"], chunk_id, seqs, exact, mm, cols)
            log.info("worker=%d saved chunk=%d rows=%d", worker_id, chunk_id, len(seqs))
        else:
            log.info("worker=%d reused chunk=%d", worker_id, chunk_id)
        done.append(chunk_id)
    log.info("worker=%d finished", worker_id)
    return done


def main():
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be >= 1.")
    log_path = args.log_path or str(Path(args.output_path).with_suffix(Path(args.output_path).suffix + ".log"))
    log = _logger(log_path)
    cfg = {
        "airr_path": args.airr_path,
        "output_path": Path(args.output_path),
        "clone_id_col": args.clone_id_col,
        "cdr3_col": args.cdr3_col,
        "locus": args.locus,
        "chain": args.chain,
        "species": args.species,
        "model_path": args.model_path,
        "is_d_present": None if args.is_d_present == "auto" else args.is_d_present == "true",
        "mirpy_path": args.mirpy_path,
        "exact_log10_pgen_col": args.exact_log10_pgen_col,
        "log10_pgen_col": args.log10_pgen_col,
        "log_path": log_path,
    }
    log.info("starting 1mm pgen calculation")
    log.info("loading AIRR in parent from %s", cfg["airr_path"])
    df, sequences, stats = _load_airr(cfg["airr_path"], cfg["clone_id_col"], cfg["cdr3_col"], cfg["locus"])
    log.info("parent loaded rows=%d", len(df))
    bounds = _chunk_bounds(len(df), args.chunk_size)
    cols = [cfg["cdr3_col"], cfg["exact_log10_pgen_col"], cfg["log10_pgen_col"]]
    done, pending = [], []
    for chunk_id, (start, end) in enumerate(bounds):
        (done if _read_chunk(cfg["output_path"], chunk_id, sequences[start:end], cols) else pending).append(
            (chunk_id, start, end)
        )
    progress = {
        **stats,
        "chain": cfg["chain"].upper(),
        "species": cfg["species"].lower(),
        "rows_total": int(len(df)),
        "workers_requested": int(args.threads),
        "workers_active": int(min(len(pending), args.threads)) if pending else 0,
        "chunk_size": int(args.chunk_size),
        "chunk_count": int(len(bounds)),
        "completed_chunks": [x[0] if isinstance(x, tuple) else x for x in done],
        "pending_chunks": [x[0] for x in pending],
    }
    _save_json(_chunk_dir(cfg["output_path"]) / "progress.json", progress)
    log.info("loaded rows=%d pending_chunks=%d workers=%d", len(df), len(pending), args.threads)
    if pending:
        jobs = [[] for _ in range(min(len(pending), args.threads))]
        for i, job in enumerate(pending):
            jobs[i % len(jobs)].append(job)
        log.info("dispatching workers=%d pending_chunks=%d", len(jobs), len(pending))
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as ex:
            futures = [ex.submit(_worker, i, job_group, cfg) for i, job_group in enumerate(jobs) if job_group]
            for fut in as_completed(futures):
                try:
                    worker_done = fut.result()
                except Exception:
                    log.exception("worker execution failed")
                    raise
                log.info("worker completed chunks=%d", len(worker_done))
                done.extend((chunk_id, None, None) for chunk_id in worker_done)
                progress["completed_chunks"] = sorted({x[0] if isinstance(x, tuple) else x for x in done})
                progress["pending_chunks"] = [i for i in range(len(bounds)) if i not in set(progress["completed_chunks"])]
                _save_json(_chunk_dir(cfg["output_path"]) / "progress.json", progress)
    exact_log = np.empty(len(df), dtype=np.float64)
    mm_log = np.empty(len(df), dtype=np.float64)
    for chunk_id, (start, end) in enumerate(bounds):
        exact_vals, mm_vals = _read_chunk(cfg["output_path"], chunk_id, sequences[start:end], cols)
        exact_log[start:end], mm_log[start:end] = exact_vals, mm_vals
    out = df.copy()
    out[args.exact_log10_pgen_col] = exact_log
    out[args.log10_pgen_col] = mm_log
    out[args.exact_pgen_col] = np.power(10.0, exact_log)
    out[args.pgen_col] = np.power(10.0, mm_log)
    cfg["output_path"].parent.mkdir(parents=True, exist_ok=True)
    suffix = cfg["output_path"].suffix.lower()
    if suffix == ".parquet":
        out.to_parquet(cfg["output_path"], index=False)
    elif suffix in {".tsv", ".airr"}:
        out.to_csv(cfg["output_path"], sep="\t", index=False)
    elif suffix == ".csv":
        out.to_csv(cfg["output_path"], index=False)
    else:
        raise ValueError(f"Unsupported output extension {suffix!r}.")
    stats_path = Path(args.output_stats_path) if args.output_stats_path else cfg["output_path"].with_suffix(cfg["output_path"].suffix + ".stats.json")
    _save_json(
        stats_path,
        {
            **progress,
            "output_path": str(cfg["output_path"].resolve()),
            "stats_path": str(stats_path.resolve()),
            "rows_written": int(len(out)),
            "model_path": str(Path(cfg["model_path"]).resolve()) if cfg["model_path"] else None,
            "mirpy_path": str(Path(cfg["mirpy_path"]).resolve()) if cfg["mirpy_path"] else None,
            "is_d_present": cfg["is_d_present"],
            "chunk_store_dir": str(_chunk_dir(cfg["output_path"]).resolve()),
            "exact_pgen_column": args.exact_pgen_col,
            "pgen_column": args.pgen_col,
            "exact_log10_pgen_column": args.exact_log10_pgen_col,
            "log10_pgen_column": args.log10_pgen_col,
        },
    )
    log.info("saved output=%s", cfg["output_path"].resolve())


if __name__ == "__main__":
    main()
