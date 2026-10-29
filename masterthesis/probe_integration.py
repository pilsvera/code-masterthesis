#!/usr/bin/env python
"""
probe_integration.py — in-sample integration-quality probe (linear head).

Measures whether batch correction encodes COMPOUND identity in a linearly readable,
batch-invariant way when all 13 Source-3 batches are corrected JOINTLY.

This is an INTEGRATION-QUALITY metric — it belongs beside silhouette / LISI / ARI.
It is NOT a deployment / test-set estimate: for transductive methods you cannot
place a genuinely new batch into this jointly-corrected space (the space, and this
head, would change on re-integration), so the head does not transfer to unseen
batches. The deployable "new batch arrives" question is Version B / LOBO-with-refit,
restricted to the inductive trio tvn/sphering/mad (see tvn_lobo.py).

What it does:
  1. All 13 batches corrected jointly upstream (make_all13_split.py + normalize_pipeline).
  2. Train a linear head (multinomial logistic regression) on the compound labels of
     the training batches.
  3. Report balanced accuracy on held-out batches — i.e. whether the biological signal
     transfers ACROSS THE BATCH AXIS *within* the integrated space.

Modes (both reported per method):
  * fixed_split -- one fixed cross-batch split (batches 8 & 11 held out from the head).
  * lobo        -- leave-one-batch-out over all 13 (mean + 95% CI + per fold); headline.

DMSO excluded (control_key="DMSO") -> head classifies the 8 compounds, chance = 1/8 =
0.125. Balanced accuracy = mean per-class recall, so chance stays 1/8 regardless of
each held-out batch's compound mix. Reuses metrics.probe_balanced_accuracy.

Usage:
  python masterthesis/probe_integration.py \
      --input-dir checkpoints/stoic-snowball-398_9llyzmze/all13/channelwise_embeddings \
      --test-batches 8 11 \
      --out-json  .../all13/channelwise_embeddings/probe_integration.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from pathlib import Path
import numpy as np
import pandas as pd

# metrics.py lives beside this file; add its dir so the import works from any CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import probe_balanced_accuracy  # noqa: E402

LABEL_KEY = "label_name"      # compound identity (biological label to classify)
BATCH_KEY = "batch"           # batch identity
CONTROL = "DMSO"              # dropped before classifying
ID_KEY = "Metadata_Sample_ID"


def _fmt(x):
    return "  nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def _ci95(vals):
    vals = np.asarray([v for v in vals if not (v is None or np.isnan(v))], dtype=float)
    if len(vals) == 0:
        return float("nan"), [float("nan"), float("nan")]
    m = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, [m - 1.96 * sem, m + 1.96 * sem]


def evaluate(Z, meta, fixed_batches, seed):
    batches = meta[BATCH_KEY].astype(str).to_numpy()
    treat = meta[LABEL_KEY].astype(str).to_numpy() != CONTROL
    out = {
        "n_compounds": int(pd.unique(meta.loc[treat, LABEL_KEY]).size),
        "n_batches": int(np.unique(batches).size),
    }

    # one fixed cross-batch split (batches held out from the HEAD, not the correction)
    if fixed_batches:
        te = np.isin(batches, [str(b) for b in fixed_batches])
        if te.any() and (~te).any():
            out["fixed_split"] = probe_balanced_accuracy(
                Z[~te], meta[~te], Z[te], meta[te],
                label_key=LABEL_KEY, control_key=CONTROL, seed=seed)
            out["fixed_test_batches"] = [str(b) for b in fixed_batches]

    # leave-one-batch-out CV (distribution / CI) — headline integration-quality number
    folds = {}
    for b in np.unique(batches):
        te = batches == b
        folds[str(b)] = probe_balanced_accuracy(
            Z[~te], meta[~te], Z[te], meta[te],
            label_key=LABEL_KEY, control_key=CONTROL, seed=seed)
    m, ci = _ci95(folds.values())
    out["lobo_mean"], out["lobo_ci95"] = m, ci
    out["lobo_folds"] = {k: (None if np.isnan(v) else round(v, 4)) for k, v in folds.items()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True,
                    help="all-13 channelwise_embeddings dir containing method subdirs")
    ap.add_argument("--test-batches", nargs="+", default=["8", "11"],
                    help="batches held out from the head for the fixed split (default: 8 11)")
    ap.add_argument("--skip-branches", nargs="*", default=[])
    ap.add_argument("--exclude-batches", nargs="*", default=[],
                    help="batch IDs dropped entirely from head train AND test "
                         "(e.g. dead batches 9 10 11 12 — no signal even within-batch)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    root = Path(args.input_dir)
    branches = sorted(
        d for d in root.iterdir()
        if d.is_dir()
        and (d / "Z_tr_corr.npy").exists() and (d / "meta_tr.pq").exists()
        and not any(s in d.name for s in args.skip_branches)
    )
    if not branches:
        raise SystemExit(f"No eligible method subdirs under {root} "
                         f"(skip={args.skip_branches}). Run stages 0+1 first.")

    print(f"{'branch':>22}  {'fixed':>6}  {'lobo':>6}  {'ci95':>16}")
    results = {}
    for d in branches:
        Z = np.load(d / "Z_tr_corr.npy")
        meta = pd.read_parquet(d / "meta_tr.pq")
        # make_all13_split leaves val==test tail rows; drop any dup Sample_IDs.
        if ID_KEY in meta.columns and meta[ID_KEY].duplicated().any():
            keep = ~meta[ID_KEY].duplicated().to_numpy()
            Z, meta = Z[keep], meta.loc[keep].reset_index(drop=True)
        if args.exclude_batches:
            live = ~meta[BATCH_KEY].astype(str).isin(
                [str(b) for b in args.exclude_batches]).to_numpy()
            Z, meta = Z[live], meta.loc[live].reset_index(drop=True)
        res = evaluate(Z, meta, args.test_batches, args.seed)
        results[d.name] = res
        lo, hi = res["lobo_ci95"]
        print(f"{d.name:>22}  {_fmt(res.get('fixed_split')):>6}  "
              f"{_fmt(res['lobo_mean']):>6}  [{_fmt(lo)},{_fmt(hi)}]")

    out_json = args.out_json or str(root / "probe_integration.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nintegration-quality probe (NOT deployment) | chance = 1/8 = 0.125 | wrote {out_json}")


if __name__ == "__main__":
    main()
