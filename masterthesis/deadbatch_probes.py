#!/usr/bin/env python
"""Dead-batch diagnosis probes — numbers only.

For every batch of a run's raw, well-aggregated `all13/none` branch:

  within  repeated stratified-half compound probe INSIDE the batch (any batch
          effect held constant; asks only whether the batch carries signal)
  cross   probe trained on the other twelve batches, tested on this one
  plate   plate-identity probe inside the batch (separates broken labels from
          empty images for the dead batches)

DMSO is excluded, so the compound probes classify 8 classes (chance 1/8).
The frozen ImageNet control rung is added via
  --runs "frozen ImageNet=<run_dir_name>".

Results are merge-cached to <out-dir>/deadbatch_diagnosis.json.

Usage:
    export PYTHONPATH=<repo root>:$PYTHONPATH
    python masterthesis/deadbatch_probes.py [--runs LABEL=run_dir ...]
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd

from masterthesis.metrics import probe_balanced_accuracy

ENCODERS = [
    ("ResNet50_Modified", "stoic-snowball-398_9llyzmze"),
    ("OpenPhenom-MAE",    "honest-snowflake-408_x79nwnqn"),
]
CONTROL = "DMSO"
LABEL_KEY = "label_name"
N_WITHIN_SPLITS = 5


def aggregate_to_well(Z, meta):
    """Collapse per-site rows to one per-well consensus profile (median Z).

    Verbatim copy of eval_all.aggregate_to_well, inlined so this script does
    not import the scanpy stack."""
    key = (["Metadata_Well_ID"] if "Metadata_Well_ID" in meta.columns
           else [c for c in ("batch", "plate", "well") if c in meta.columns])
    if not key:
        return Z, meta
    groups = meta.groupby(key, sort=False).indices
    Z_out = np.empty((len(groups), Z.shape[1]), dtype=np.float32)
    first_rows = np.empty(len(groups), dtype=np.int64)
    for i, idx in enumerate(groups.values()):
        Z_out[i] = np.median(Z[idx], axis=0)
        first_rows[i] = idx[0]
    return Z_out, meta.iloc[first_rows].reset_index(drop=True)


def _load_raw_wellagg(repo: Path, run: str):
    d = repo / "checkpoints" / run / "all13" / "channelwise_embeddings" / "none"
    Z = np.load(d / "Z_tr_corr.npy")
    meta = pd.read_parquet(d / "meta_tr.pq")
    n0 = len(meta)
    Z, meta = aggregate_to_well(Z, meta)
    print(f"    {run}: {n0} sites -> {len(meta)} wells, dim {Z.shape[1]}")
    return Z, meta


def _within_batch_acc(Z, meta, mask, seed_base=0):
    from sklearn.model_selection import StratifiedShuffleSplit

    Zb, mb = Z[mask], meta[mask].reset_index(drop=True)
    keep = mb[LABEL_KEY].astype(str) != CONTROL
    Zb, mb = Zb[keep.to_numpy()], mb[keep].reset_index(drop=True)
    y = mb[LABEL_KEY].astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    accs = []
    sss = StratifiedShuffleSplit(n_splits=N_WITHIN_SPLITS, test_size=0.5,
                                 random_state=42 + seed_base)
    for tr, te in sss.split(Zb, y):
        a = probe_balanced_accuracy(Zb[tr], mb.iloc[tr], Zb[te], mb.iloc[te],
                                    label_key=LABEL_KEY, control_key=None, seed=42)
        if not np.isnan(a):
            accs.append(a)
    if not accs:
        return float("nan"), float("nan")
    return float(np.mean(accs)), float(np.std(accs))


def _plate_within_batch(Z, meta, mask):
    from sklearn.model_selection import StratifiedShuffleSplit

    Zb, mb = Z[mask], meta[mask].reset_index(drop=True)
    if "plate" not in mb.columns:
        return float("nan"), float("nan")
    y = mb["plate"].astype(str).to_numpy()
    counts = pd.Series(y).value_counts()
    keep = pd.Series(y).isin(counts[counts >= 4].index).to_numpy()
    Zb, mb, y = Zb[keep], mb[keep].reset_index(drop=True), y[keep]
    n_cls = len(np.unique(y))
    if n_cls < 2:
        return float("nan"), float("nan")

    accs = []
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=42)
    for tr, te in sss.split(Zb, y):
        a = probe_balanced_accuracy(Zb[tr], mb.iloc[tr], Zb[te], mb.iloc[te],
                                    label_key="plate", control_key=None, seed=42)
        if not np.isnan(a):
            accs.append(a)
    return (float(np.mean(accs)) if accs else float("nan")), 1.0 / n_cls


def compute(repo: Path, cache: Path, recompute: bool, encoders=None) -> dict:
    # MERGE semantics: the cache keeps every rung ever computed; a partial run
    # never drops the rungs it was not asked for.
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    wanted = list(encoders or ENCODERS)
    if not recompute and all(label in cached for label, _ in wanted):
        print(f"  using cache {cache}")
        return cached

    out = dict(cached)
    for label, run in wanted:
        if label in cached and not recompute:
            print(f"  [{label}] from cache")
            continue
        print(f"  [{label}]")
        Z, meta = _load_raw_wellagg(repo, run)
        batches = sorted(meta["batch"].unique().tolist())
        rows = {}
        for b in batches:
            m = (meta["batch"] == b).to_numpy()
            within, sd = _within_batch_acc(Z, meta, m)
            cross = probe_balanced_accuracy(
                Z[~m], meta[~m], Z[m], meta[m],
                label_key=LABEL_KEY, control_key=CONTROL, seed=42)
            plate_acc, plate_chance = _plate_within_batch(Z, meta, m)
            rows[str(int(b))] = dict(within=within, within_sd=sd,
                                     cross=float(cross), n_wells=int(m.sum()),
                                     plate=plate_acc, plate_chance=plate_chance)
            print(f"    batch {int(b):2d}  within {within:.3f}±{sd:.3f}   "
                  f"cross {cross:.3f}   plate {plate_acc:.3f} "
                  f"(chance {plate_chance:.3f})   n={int(m.sum())}")
        out[label] = rows

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cache}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parents[2],
                    help="directory holding checkpoints/")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--runs", nargs="*", default=[],
                    help='extra rungs as "LABEL=run_dir_name" (appended to ENCODERS)')
    ap.add_argument("--only-runs", action="store_true",
                    help="use ONLY --runs, skipping the two default encoders")
    args = ap.parse_args()
    out_dir = args.out_dir or (args.repo_root / "thesis_figures/deadbatch")
    cache = args.cache or (out_dir / "deadbatch_diagnosis.json")

    extra = []
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f'--runs entry must be LABEL=run_dir_name, got {spec!r}')
        label, run = spec.split("=", 1)
        extra.append((label, run))
    encoders = (extra if args.only_runs else [*ENCODERS, *extra]) or ENCODERS

    compute(args.repo_root, cache, args.recompute, encoders)


if __name__ == "__main__":
    main()
