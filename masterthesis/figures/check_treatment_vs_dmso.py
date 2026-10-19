#!/usr/bin/env python
"""Is a treated well still distinguishable from a control well after correction?

Every biological metric in metrics.py asks one question: do replicates of the same
compound stay together. None asks whether a treated well can be told apart from
DMSO at all. That is the collapse failure mode — a correction that
pulls the treatments into the control cloud destroys biology while leaving replicate
coherence partly intact, because the replicates can collapse together.

Four readouts per branch, on the well-aggregated profiles:

  auroc_vs_dmso   5-fold CV logistic regression separating each compound's wells
                  from DMSO wells, averaged over the eight compounds (per-compound
                  values written separately). No extra PCA step: these branches
                  are already the 50-D projection.
  lisi_ctrl       LISI with "control vs treatment" as the label. 1 = treatments
                  occupy their own neighbourhoods, 2 = fully mixed with controls.
                  The mirror of the treatments-only cLISI repair.
  activity_map    mAP where the negatives are control wells on the same plate
                  rather than other compounds — the JUMP "is this compound active"
                  flavour, as opposed to "does it retrieve its own replicates".
  maha_ratio      median Mahalanobis distance of treated wells to the DMSO centroid,
                  in the DMSO covariance, divided by the controls' own median. The
                  classical induction score, made scale-free so branches compare.
                  READ WITH CARE: TVN's sphering normalises exactly this quantity,
                  so this readout is structurally unfavourable to TVN by construction.

None of these is "higher is better" on its own. A method can push treatments away
from the controls without keeping replicates together, the same trap as iLISI on the
batch axis, so read them next to the replicate metrics.

Usage:
    conda activate mm_cpu
    PYTHONPATH=. python masterthesis/figures/check_treatment_vs_dmso.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "masterthesis"))
import copairs.map as copairs_map                                        # noqa: E402
from masterthesis import metrics as M                                    # noqa: E402
from masterthesis.eval_all import (load_npz_with_meta, pca_reduce,       # noqa: E402
                                   aggregate_to_well)

RUN_DIRS = {
    12: REPO / "checkpoints/ResNet_source_3_channelwise/eager-sound-321_dgpjedln/channelwise_embeddings",
    24: REPO / "checkpoints/silver-planet-397_vbjalfjr/channelwise_embeddings",
    42: REPO / "checkpoints/stoic-snowball-398_9llyzmze/channelwise_embeddings",
}
META = REPO / "source_3_w_neg.pq"
CTRL = "DMSO"
METHODS = ["raw", "tvn", "harmony", "combat", "seurat", "mnn", "fastmnn", "bbknn", "scvi"]
# Controls outnumber any single compound ~25:1. class_weight="balanced" handles the
# imbalance and AUROC is insensitive to it, but capping keeps 216 model fits tractable.
MAX_CTRL = 5000


def _norm_meta(meta):
    ren = {"Metadata_Batch": "batch", "Metadata_Plate": "plate", "Metadata_Well": "well"}
    meta = meta.rename(columns={k: v for k, v in ren.items() if k in meta.columns})
    if "label_name" not in meta.columns and "pert_iname_str" in meta.columns:
        meta = meta.rename(columns={"pert_iname_str": "label_name"})
    return meta


def load_branch(ce: Path, method: str, src: pd.DataFrame):
    if method == "raw":
        Xtr, mtr = load_npz_with_meta(ce / "train_channelwise_embeddings.npz", src, CTRL)
        Xva, mva = load_npz_with_meta(ce / "validation_channelwise_embeddings.npz", src, CTRL)
        Xall = np.vstack([Xtr, Xva])
        meta = pd.concat([mtr, mva], ignore_index=True)
        (_, Z), _, _ = pca_reduce(Xtr, [Xtr, Xall], 50)
    else:
        Z = np.load(ce / method / "Z_tr_corr.npy")
        meta = pd.read_parquet(ce / method / "meta_tr.pq")
    return aggregate_to_well(Z, _norm_meta(meta))


def auroc_vs_dmso(Z, labels, seed=42):
    """Per-compound cross-validated separability from the controls."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    ctrl_idx = np.where(labels == CTRL)[0]
    if len(ctrl_idx) > MAX_CTRL:
        ctrl_idx = rng.choice(ctrl_idx, size=MAX_CTRL, replace=False)
    out = {}
    for c in sorted(set(labels) - {CTRL}):
        idx = np.concatenate([np.where(labels == c)[0], ctrl_idx])
        X, y = Z[idx], (labels[idx] == c).astype(int)
        if y.sum() < 5:
            out[c] = np.nan
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced"))
        out[c] = float(cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean())
    return out


def lisi_ctrl(Z, meta):
    m = meta.copy()
    m["ctrl"] = np.where(m["label_name"].astype(str) == CTRL, "control", "treatment")
    return M.compute_lisi(Z, m, ["batch"], label_key="ctrl")["lisi_ctrl"]


def activity_map(Z, meta):
    """mAP against control negatives: positives are same-compound wells, negatives
    are control wells on the same plate (rather than any other compound)."""
    m = meta[["label_name", "plate"]].astype(str).copy().reset_index(drop=True)
    is_ctrl = (m["label_name"] == CTRL).to_numpy()
    m["ctrl"] = np.where(is_ctrl, "control", "treatment")
    m.loc[is_ctrl, "label_name"] = [f"{CTRL}_{i}" for i in range(int(is_ctrl.sum()))]
    ap = copairs_map.average_precision(
        m, Z,
        pos_sameby=["label_name"], pos_diffby=[],
        neg_sameby=["plate"], neg_diffby=["label_name", "ctrl"],
        batch_size=20000, distance="cosine", progress_bar=False)
    ap = ap[~is_ctrl].copy()
    ap = ap[ap["n_pos_pairs"] > 0]
    if ap.empty:
        return float("nan")
    mp = copairs_map.mean_average_precision(
        ap, sameby=["label_name"], null_size=10000, threshold=0.05, seed=42,
        progress_bar=False)
    return round(float(mp["mean_average_precision"].mean()), 4)


def maha_ratio(Z, labels):
    """Median Mahalanobis distance of treated wells to the DMSO centroid, in the
    DMSO covariance, over the controls' own median distance."""
    C = Z[labels == CTRL]
    mu = C.mean(axis=0)
    S = np.cov(C, rowvar=False) + 1e-6 * np.eye(Z.shape[1])
    Si = np.linalg.inv(S)

    def med(A):
        D = A - mu
        return float(np.median(np.sqrt(np.einsum("ij,jk,ik->i", D, Si, D))))

    ref = med(C)
    treat = med(Z[labels != CTRL])
    return round(treat / ref, 4) if ref > 0 else float("nan")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    out_dir = REPO / "thesis_figures" / "metric_concordance"
    ap_.add_argument("--out", type=Path, default=out_dir / "treatment_vs_dmso.csv")
    ap_.add_argument("--out-per-compound", type=Path,
                     default=out_dir / "treatment_vs_dmso_per_compound.csv")
    a = ap_.parse_args()

    src = pd.read_parquet(META)
    src["Metadata_Sample_ID"] = src["Metadata_Sample_ID"].astype(str)

    rows, per_cmp = [], []
    for seed, ce in RUN_DIRS.items():
        for m in METHODS:
            if m != "raw" and not (ce / m / "Z_tr_corr.npy").exists():
                print(f"[seed {seed}][{m}] missing — skipped", flush=True)
                continue
            Z, meta = load_branch(ce, m, src)
            labels = meta["label_name"].astype(str).values

            au = auroc_vs_dmso(Z, labels)
            for c, v in au.items():
                per_cmp.append(dict(seed=seed, method=m, compound=c, auroc_vs_dmso=v))
            r = dict(seed=seed, method=m,
                     auroc_vs_dmso=round(float(np.nanmean(list(au.values()))), 4),
                     lisi_ctrl=round(lisi_ctrl(Z, meta), 4),
                     activity_map=activity_map(Z, meta),
                     maha_ratio=maha_ratio(Z, labels),
                     n_wells=len(meta))
            rows.append(r)
            print(f"[seed {seed}][{m:8s}] auroc-vs-DMSO {r['auroc_vs_dmso']:.3f}  "
                  f"LISI(ctrl) {r['lisi_ctrl']:.3f}  activity mAP {r['activity_map']}  "
                  f"maha ratio {r['maha_ratio']}", flush=True)

    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.DataFrame(per_cmp).to_csv(a.out_per_compound, index=False)
    print(f"\n[wrote] {a.out}\n[wrote] {a.out_per_compound}")

    cols = ["auroc_vs_dmso", "lisi_ctrl", "activity_map", "maha_ratio"]
    g = df.groupby("method")[cols]
    mean, sd = g.mean(), g.std(ddof=1)
    mean = mean.sort_values("auroc_vs_dmso", ascending=False)
    print("\nseed means (sorted by compound-vs-DMSO AUROC):")
    for m in mean.index:
        print("  " + f"{m:8s}" + "  ".join(
            f" {c} {mean.loc[m, c]:.3f}±{sd.loc[m, c]:.3f}" for c in cols))


if __name__ == "__main__":
    main()
