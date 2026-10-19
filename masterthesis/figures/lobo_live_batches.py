#!/usr/bin/env python
"""Leave-one-batch-out summaries restricted to the batches that carry signal.

tvn_lobo.py stores one row per (held-out batch, method) with every metric, so
the dead-batch restriction is a filter on that CSV, not a rerun.
Batches 9-12 of Source 3 carry no decodable compound signal even when the probe
is trained and tested inside them, so an absolute mean over all 13 folds averages
a real measurement with four folds at chance. The paired delta is fold-matched
and therefore survives the dead folds; the absolute level does not, which is why
the absolute probe accuracy is only meaningful on the live folds.

Source 1 has no dead batches, so its live and all-batch rows coincide; they are
printed anyway as a check.

Reports, per encoder x source: the absolute balanced accuracy and the paired
delta from raw with a percentile-bootstrap CI over held-out batches, for the
all-batch and live-batch fold sets side by side.

NOTE the CSVs carry two probe columns: ``bacc`` is the nine-way probe (DMSO
kept, chance 1/9); ``bacc_xdmso`` drops DMSO from both sides (eight-way, chance
1/8) and is the variant the thesis reports, directly comparable with the
integration probe of probe_integration.py.

Usage:
    conda activate mm_cpu
    PYTHONPATH=. python masterthesis/figures/lobo_live_batches.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LOBO = REPO / "thesis_figures" / "tvn_holdout"
DEAD_S3 = {9, 10, 11, 12}          # dead batches from the batch quality control
METHODS = ["tvn", "sphering", "mad"]
N_BOOT = 5000
SEED = 0

SETS = {
    "ResNet S3": (LOBO / "lobo" / "tvn_lobo_per_batch.csv", DEAD_S3),
    "MAE S3":    (LOBO / "lobo_mae" / "tvn_lobo_per_batch.csv", DEAD_S3),
    "ResNet S1": (LOBO / "lobo_s1" / "eager-snowball-414_wrdhgrqx"
                  / "tvn_lobo_per_batch.csv", set()),
    "MAE S1":    (LOBO / "lobo_s1" / "generous-pond-415_gjyqp3kt"
                  / "tvn_lobo_per_batch.csv", set()),
}


def boot_ci(v, rng, n=N_BOOT):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="bacc",
                    help="column of tvn_lobo_per_batch.csv (bacc, auroc, ilisi, kbet, sil)")
    ap.add_argument("--out", type=Path,
                    default=LOBO / "lobo_live_batches.csv")
    a = ap.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []
    for name, (path, dead) in SETS.items():
        if not path.exists():
            print(f"[skip] {name}: {path} missing")
            continue
        df = pd.read_csv(path)
        df["batch"] = df["batch"].astype(int)
        print(f"\n=== {name}  ({a.metric}) ===")
        for tag, sub in (("all", df), ("live", df[~df.batch.isin(dead)])):
            piv = sub.pivot(index="batch", columns="method", values=a.metric)
            if "raw" not in piv.columns:
                continue
            line = f"  {tag:4s} n={len(piv):2d}  raw {piv['raw'].mean():.3f}  |"
            for m in METHODS:
                if m not in piv.columns:
                    continue
                d = (piv[m] - piv["raw"]).dropna()
                lo, hi = boot_ci(d, rng)
                line += (f"  {m}: abs {piv[m].mean():.3f} "
                         f"delta {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
                rows.append(dict(dataset=name, folds=tag, n=len(piv), metric=a.metric,
                                 method=m, abs_raw=piv["raw"].mean(),
                                 abs_method=piv[m].mean(), delta=d.mean(),
                                 ci_lo=lo, ci_hi=hi))
            print(line)

    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"\n[wrote] {a.out}")


if __name__ == "__main__":
    main()
