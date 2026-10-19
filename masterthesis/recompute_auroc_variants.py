#!/usr/bin/env python
"""Add the corrected replicate-AUROC variants to evaluations that predate them.

metrics.evaluate_branch gained two readouts after several runs had already been
evaluated, so their eval_all_wellagg.json carries only the default ``auroc``:

    auroc_nodmso   DMSO dropped from both pair pools
    auroc_xbatch   DMSO dropped AND positives restricted to different batches

The default counts DMSO self-coherence and within-batch pairs. With DMSO filling
about three quarters of the wells it supplies most of the different-compound
pairs, so the default largely reports whether a well is treated at all rather
than which compound it holds; and two wells of one compound inside one batch
have no batch effect between them, which is a free mark for a correction that
never had to remove one. The chapter therefore ranks on auroc_xbatch, and every
run it draws on has to carry it.

This script recomputes only those two keys, with the same calls
metrics.evaluate_branch makes (metrics.py:691-694) on the same well-aggregated
profiles, and writes them back under the same names. Every other metric in the
file is left untouched, so it is far cheaper than a full re-evaluation and
cannot perturb LISI, kBET, Leiden or mAP.

Branches are reconstructed exactly as eval_all does:
  raw            train + validation embeddings, PCA fit on train, then aggregated
  every other    Z_tr_corr.npy + meta_tr.pq (already reduced), then aggregated

It is idempotent: a branch that already holds both keys is skipped unless
--force is given, and each JSON is written atomically so a killed job cannot
leave a truncated file behind.

Usage:
    python masterthesis/recompute_auroc_variants.py [--runs <embeddings-dir> ...]

Note the discovery glob walks the whole checkpoints tree, which is slow when the
repository sits on a mounted filesystem; --list is the cheap way to see the work
list, and passing --runs explicitly skips discovery altogether.

Usage (direct):
    conda activate mm_cuda          # mm_cpu locally
    export PYTHONPATH=$PWD
    python masterthesis/recompute_auroc_variants.py --list
    python masterthesis/recompute_auroc_variants.py
    python masterthesis/recompute_auroc_variants.py \
        --runs checkpoints/sandy-surf-401_r08dridp/channelwise_embeddings
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "masterthesis"))

from masterthesis import metrics as M                                    # noqa: E402
from masterthesis.eval_all import (aggregate_to_well, load_npz_with_meta,  # noqa: E402
                                   pca_reduce)

CTRL = "DMSO"
KEYS = ("auroc_nodmso", "auroc_xbatch")


def discover(root: Path, json_name: str) -> list[Path]:
    """Every embeddings dir under `root` holding an evaluation of that name."""
    return sorted(p.parent for p in root.glob(f"**/{json_name}"))


def branch_needs_work(branch: dict, force: bool) -> bool:
    return force or any(k not in branch for k in KEYS)


def load_raw(emb_dir: Path, meta_path: Path, n_pca: int, aggregate: bool):
    """Rebuild the raw branch the way eval_all does (eval_all.py:415-432)."""
    src = pd.read_parquet(meta_path)
    src["Metadata_Sample_ID"] = src["Metadata_Sample_ID"].astype(str)
    X_tr, meta_tr = load_npz_with_meta(
        emb_dir / "train_channelwise_embeddings.npz", src, CTRL)
    X_va, meta_va = load_npz_with_meta(
        emb_dir / "validation_channelwise_embeddings.npz", src, CTRL)
    X_all = np.vstack([X_tr, X_va])
    meta = pd.concat([meta_tr, meta_va], ignore_index=True)
    (_, Z), _, _ = pca_reduce(X_tr, [X_tr, X_all], n_pca)
    return aggregate_to_well(Z, meta) if aggregate else (Z, meta)


def load_branch(emb_dir: Path, sub: str, aggregate: bool):
    Z = np.load(emb_dir / sub / "Z_tr_corr.npy")
    meta = pd.read_parquet(emb_dir / sub / "meta_tr.pq")
    return aggregate_to_well(Z, meta) if aggregate else (Z, meta)


def variants(Z, meta) -> dict:
    """The two corrected readouts, exactly as metrics.evaluate_branch computes them."""
    out = {"auroc_nodmso": M.auroc_replicates(Z, meta, control_key=CTRL)}
    if "batch" in meta.columns:
        out["auroc_xbatch"] = M.auroc_replicates(
            Z, meta, control_key=CTRL, cross_batch=True, batch_key="batch")
    return out


def write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="*",
                    help="embeddings dirs to process; default: auto-discover under checkpoints/")
    ap.add_argument("--root", type=Path, default=REPO / "checkpoints")
    ap.add_argument("--json-name", default="eval_all_wellagg.json")
    ap.add_argument("--meta-path", type=Path, default=REPO / "source_3_w_neg.pq",
                    help="source metadata, needed only to rebuild the raw branch")
    ap.add_argument("--n-pca", type=int, default=50)
    ap.add_argument("--no-aggregate", action="store_true",
                    help="site-level evaluation (use with --json-name eval_all.json)")
    ap.add_argument("--skip-raw", action="store_true",
                    help="leave the raw branch alone (it is the only one needing --meta-path)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute everything but write nothing")
    ap.add_argument("--list", action="store_true",
                    help="only report which branches are missing; loads no embeddings")
    a = ap.parse_args()

    aggregate = not a.no_aggregate
    dirs = a.runs or discover(a.root, a.json_name)
    print(f"[scan] {len(dirs)} evaluation(s) named {a.json_name}", flush=True)

    total_done = total_skipped = total_failed = 0
    for emb_dir in dirs:
        fp = emb_dir / a.json_name
        data = json.loads(fp.read_text())
        todo = [k for k, b in data.items()
                if isinstance(b, dict) and branch_needs_work(b, a.force)]
        if a.skip_raw:
            todo = [k for k in todo if k != "raw"]
        if not todo:
            print(f"[ok  ] {emb_dir.parent.name}: nothing missing", flush=True)
            continue
        print(f"[run ] {emb_dir.parent.name}: {len(todo)} of {len(data)} branches "
              f"-> {', '.join(todo)}", flush=True)
        if a.list:
            total_skipped += len(todo)
            continue

        changed = False
        for name in todo:
            t0 = time.time()
            try:
                if name == "raw":
                    if not a.meta_path.exists():
                        raise FileNotFoundError(f"{a.meta_path} (needed for raw)")
                    Z, meta = load_raw(emb_dir, a.meta_path, a.n_pca, aggregate)
                else:
                    Z, meta = load_branch(emb_dir, name, aggregate)
                got = variants(Z, meta)
            except Exception as exc:                       # noqa: BLE001
                total_failed += 1
                print(f"       {name:22s} FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            data[name].update(got)
            changed = True
            total_done += 1
            vals = "  ".join(f"{k.split('_')[-1]} {v['auroc_replicating']:.4f}"
                             for k, v in got.items())
            print(f"       {name:22s} {Z.shape[0]:6d} rows  {vals}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

        if changed and not a.dry_run:
            write_atomic(fp, data)
            print(f"       [wrote] {fp}", flush=True)
        elif changed:
            print("       [dry-run] not written", flush=True)

    print(f"\n[done] {total_done} branches updated, {total_skipped} skipped, "
          f"{total_failed} failed", flush=True)
    sys.exit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
