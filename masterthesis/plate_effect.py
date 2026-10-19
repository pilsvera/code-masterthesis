#!/usr/bin/env python
"""Plate-level technical effect below the batch (DMSO wells only) — numbers only.

Produces every number of the plate-effect subsection and the corrected-branch
table of the thesis:

(1) nested variance decomposition of the DMSO well profiles in the shared
    50-PC space:  batch  /  plate-within-batch  /  residual (well)
(2) plate-identity probe: within each batch, logistic regression predicting
    the plate of a DMSO well (5-fold CV, balanced accuracy, reported as
    multiples of the 1/n_plates chance level)
(3) the same readouts on the corrected branches (combat, tvn, harmony, mnn)

Output: <out-dir>/plate_effect.json

Usage:
    export PYTHONPATH=<repo root>:$PYTHONPATH
    python masterthesis/plate_effect.py [--encoders resnet mae] [--repo-root ...]
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from masterthesis.common import load_split, scale_and_pca
from masterthesis.eval_all import aggregate_to_well

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RUNS = {
    "resnet": "checkpoints/stoic-snowball-398_9llyzmze/channelwise_embeddings",
    "mae":    "checkpoints/honest-snowflake-408_x79nwnqn/channelwise_embeddings",
}
BRANCHES = ["combat", "tvn", "harmony", "mnn"]


def dmso_wells(Z, meta):
    Zw, mw = aggregate_to_well(Z, meta)
    dm = mw["is_control"].values
    return Zw[dm], mw[dm].reset_index(drop=True)


def decompose(Zd, md):
    g = Zd - Zd.mean(0)
    tot = (g ** 2).sum()
    bm = np.zeros_like(g)
    for _, idx in md.groupby("batch").indices.items():
        bm[idx] = g[idx].mean(0)
    pm = np.zeros_like(g)
    for _, idx in md.groupby(["batch", "plate"]).indices.items():
        pm[idx] = g[idx].mean(0)
    return {"batch": float((bm ** 2).sum() / tot),
            "plate": float(((pm - bm) ** 2).sum() / tot),
            "well":  float(((g - pm) ** 2).sum() / tot)}


def plate_probe(Zd, md):
    """Per-batch balanced accuracy of a plate classifier, as multiple of chance."""
    out = {}
    for b, sub in md.groupby("batch"):
        idx = sub.index.values
        y = sub["plate"].values
        n = len(set(y))
        if n < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        yp = cross_val_predict(clf, Zd[idx], y,
                               cv=StratifiedKFold(5, shuffle=True, random_state=0))
        out[str(b)] = {"n_plates": n,
                       "bacc": float(balanced_accuracy_score(y, yp)),
                       "x_chance": float(balanced_accuracy_score(y, yp) * n)}
    return out


def batch_probe(Zd, md):
    y = md["batch"].values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    yp = cross_val_predict(clf, Zd, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    n = md["batch"].nunique()
    return {"bacc": float(balanced_accuracy_score(y, yp)),
            "x_chance": float(balanced_accuracy_score(y, yp) * n)}


def load_raw(ce: Path, repo: Path):
    meta_all = pd.read_parquet(repo / "source_3_w_neg.pq")
    X_tr, m_tr = load_split(ce / "train_channelwise_embeddings.npz", meta_all, "DMSO")
    X_va, m_va = load_split(ce / "validation_channelwise_embeddings.npz", meta_all, "DMSO")
    Z_tr, Z_va, _, _, _, _ = scale_and_pca(X_tr, X_va, X_va[:1], 50)
    return np.vstack([Z_tr, Z_va]), pd.concat([m_tr, m_va], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=["resnet", "mae"])
    p.add_argument("--repo-root", type=Path,
                   default=Path(__file__).resolve().parents[2],
                   help="directory holding checkpoints/ and source_3_w_neg.pq")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    repo = args.repo_root
    out_dir = args.out_dir or (repo / "thesis_figures/plate_effect")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for enc in args.encoders:
        ce = repo / RUNS[enc]
        print(f"[{enc}] raw ...", flush=True)
        Zd, md = dmso_wells(*load_raw(ce, repo))
        results[enc] = {"raw": {"var": decompose(Zd, md),
                                "plate_probe": plate_probe(Zd, md),
                                "batch_probe": batch_probe(Zd, md),
                                "n_dmso_wells": int(len(md)),
                                "n_plates": int(md["plate"].nunique())}}
        for br in BRANCHES:
            d = ce / br
            if not (d / "Z_tr_corr.npy").exists():
                print(f"  [skip] {enc}/{br}")
                continue
            print(f"[{enc}] {br} ...", flush=True)
            Z = np.load(d / "Z_tr_corr.npy")
            meta = pd.read_parquet(d / "meta_tr.pq")
            Zd2, md2 = dmso_wells(Z, meta)
            pr = plate_probe(Zd2, md2)
            results[enc][br] = {"var": decompose(Zd2, md2),
                                "plate_probe_mean_x_chance":
                                    float(np.mean([v["x_chance"] for v in pr.values()])),
                                "batch_probe": batch_probe(Zd2, md2)}
        raw_pr = results[enc]["raw"]["plate_probe"]
        results[enc]["raw"]["plate_probe_mean_x_chance"] = \
            float(np.mean([v["x_chance"] for v in raw_pr.values()]))

    out = out_dir / "plate_effect.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
