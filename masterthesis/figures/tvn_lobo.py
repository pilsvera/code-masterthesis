#!/usr/bin/env python
"""Leave-one-batch-out (LOBO) generalization of the deployable corrections.

For EACH of the 13 batches b,

    R = the other 12 batches  (reference / "the batches you already have")
    H = batch b               (the "new, unseen batch")

  * fit TVN's recipe (scaler + feature-PCA + target_cov) on R's DMSO ONLY,
  * fit the final 50-D PCA on R ONLY,
  * transform H using H's OWN DMSO controls (never fit on H),
  * score H ONLY, vs a RAW baseline that is also R-fit-PCA projected onto H.

This yields 13 paired (TVN - raw) deltas per metric; reported as mean + bootstrap CI
OVER BATCHES (the batch is the experimental unit). Lead metrics:
  bio : AUROC replicate-vs-null on H   (higher = biology preserved)
  mix : iLISI(H-vs-R) -> 2, silhouette(H-vs-R) -> 0, kBET-rej -> 0  (H integrates)

"Better" (pre-registered): iLISI improves over raw AND AUROC does NOT drop below raw.
A large negative Δ AUROC alongside big mixing gains = the over-mixing trap.

Extension: CORRECTORS is a dict; add sphering/MAD correctors with the same
(X_R, meta_R, X_H, meta_H, batch_key) -> (Z_R, Z_H) signature to bake them off here.

Runtime: ~13 x (full PCA on R DMSO + transform ~280k sites). This is a LONG job —
prefer the cluster (sbatch) over the login node. Use --batches 0 9 8 for a 3-fold smoke
test first, and --dmso-subsample 2000 to speed the per-fold fit.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import numpy.linalg as _la
import pandas as pd

# econ-SVD guard: Arevalo's Spherize calls svd(full_matrices=True) -> a 145k x 145k
# U on the DMSO matrix (158 GiB). For n>=d the economy SVD gives the identical
# Vt/Sigma (hence identical whitening) without the blow-up. Harmless for TVN/MAD.
_orig_svd = _la.svd
def _econ_svd(a, full_matrices=True, **kw):
    return _orig_svd(a, full_matrices=False, **kw)
_la.svd = _econ_svd
np.linalg.svd = _econ_svd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "masterthesis"))   # tvn.py imports by bare name
from masterthesis.common import load_split, scale_and_pca  # noqa
from masterthesis import metrics as M  # noqa
from masterthesis.tvn import TVN  # noqa
from sklearn.decomposition import PCA  # noqa

CE = REPO / "checkpoints/stoic-snowball-398_9llyzmze/channelwise_embeddings"
META_PATH = REPO / "source_3_w_neg.pq"
OUT = REPO / "thesis_figures" / "tvn_holdout" / "lobo"
BATCH_KEY = "batch"
N_COMP = 50
N_REF_SUB = 2500     # reference wells subsampled for the mixing metrics
SEED = 42


# --------------------------------------------------------------------------- #
# Well aggregation (inlined from eval_all.py to avoid its TF/copairs import)
# --------------------------------------------------------------------------- #
def _well_key_cols(meta):
    if "Metadata_Well_ID" in meta.columns:
        return ["Metadata_Well_ID"]
    return [c for c in ("batch", "plate", "well") if c in meta.columns]


def aggregate_to_well(Z, meta):
    key = _well_key_cols(meta)
    if not key:
        return Z, meta
    groups = meta.groupby(key, sort=False).indices
    n = len(groups)
    Z_out = np.empty((n, Z.shape[1]), dtype=np.float32)
    first = np.empty(n, dtype=np.int64)
    for i, idx in enumerate(groups.values()):
        Z_out[i] = np.median(Z[idx], axis=0)
        first[i] = idx[0]
    return Z_out, meta.iloc[first].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Correctors: (X_R, meta_R, X_H, meta_H, batch_key) -> (Z_R, Z_H) in R-fit space
# --------------------------------------------------------------------------- #
def correct_raw(X_R, meta_R, X_H, meta_H, batch_key, dmso_subsample=None):
    Z_R, Z_H, _, _, _, _ = scale_and_pca(X_R, X_H, X_H, N_COMP, seed=SEED)
    return Z_R, Z_H


def correct_tvn(X_R, meta_R, X_H, meta_H, batch_key, dmso_subsample=None):
    ctrl_R = meta_R["is_control"].values
    Xneg_R, bneg_R = X_R[ctrl_R], meta_R.loc[ctrl_R, batch_key].to_numpy()
    ctrl_H = meta_H["is_control"].values
    Xneg_H, bneg_H = X_H[ctrl_H], meta_H.loc[ctrl_H, batch_key].to_numpy()
    if ctrl_H.sum() == 0:
        raise ValueError("held-out batch has no DMSO controls — TVN cannot transform it")

    tvn = TVN(pcafeat=True)
    tvn.fit(Xneg_R, bneg_R, dmso_subsample=dmso_subsample, seed=SEED)
    X_R_tvn = tvn.transform(X_R, meta_R[batch_key].to_numpy(), Xneg_R, bneg_R)
    X_H_tvn = tvn.transform(X_H, meta_H[batch_key].to_numpy(), Xneg_H, bneg_H)

    n_comp = min(N_COMP, X_R_tvn.shape[1], X_R_tvn.shape[0])
    pca = PCA(n_components=n_comp, random_state=SEED)
    Z_R = pca.fit_transform(X_R_tvn)
    Z_H = pca.transform(X_H_tvn)
    return Z_R, Z_H


def correct_mad(X_R, meta_R, X_H, meta_H, batch_key, dmso_subsample=None):
    """Per-batch MAD: each batch (incl. the held-out one) self-normalises by its
    OWN DMSO — no train fit, the most deployment-trivial correction. PCA fit on R."""
    from masterthesis.mad import normalize as mad_normalize
    XR = mad_normalize(X_R, meta_R, batch_key).astype(np.float32)
    XH = mad_normalize(X_H, meta_H, batch_key).astype(np.float32)
    Z_R, Z_H, _, _, _, _ = scale_and_pca(XR, XH, XH, N_COMP, seed=SEED)
    return Z_R, Z_H


def correct_sphering(X_R, meta_R, X_H, meta_H, batch_key, dmso_subsample=None):
    """Global ZCA whitening (W fit on R's DMSO), applied blindly to H. PCA fit on R."""
    from masterthesis.sphering import apply as sphering_apply
    XR, XH, _ = sphering_apply(X_R, X_H, X_H, meta_R, method="ZCA-cor", eps=1e-6)
    Z_R, Z_H, _, _, _, _ = scale_and_pca(XR, XH, XH, N_COMP, seed=SEED)
    return Z_R, Z_H


CORRECTORS = {"raw": correct_raw, "tvn": correct_tvn,
              "sphering": correct_sphering, "mad": correct_mad}


# --------------------------------------------------------------------------- #
# Per-held-out-batch metrics (well level)
# --------------------------------------------------------------------------- #
def score_holdout(Z_R, meta_R, Z_H, meta_H):
    """Bio (AUROC) + integration of H into a subsample of R.

    Two replicate AUROCs are reported. `auroc` is the legacy readout: pairs
    within the held-out batch only, DMSO included — necessarily within-batch, so
    it never crosses the boundary the correction was fitted across. `auroc_xref`
    is the chapter's replicate definition applied to this protocol: positives are
    same-compound pairs between the held-out batch and the corrected reference,
    negatives cross-compound pairs across the same boundary, DMSO dropped from
    both sides. Every `auroc_xref` pair crosses a batch boundary by construction.
    """
    Z_Hw, meta_Hw = aggregate_to_well(Z_H, meta_H)
    Z_Rw, meta_Rw = aggregate_to_well(Z_R, meta_R)
    auroc = M.auroc_replicates(Z_Hw, meta_Hw)["auroc_replicating"]
    auroc_xref = M.auroc_replicates_anchor(Z_Hw, meta_Hw, Z_Rw, meta_Rw,
                                           control_key="DMSO")["auroc_replicating"]
    # cross-batch linear-probe balanced accuracy: fit on the reference (the batches
    # you already have), predict the held-out batch — does the correction keep the
    # compounds classifiable on an unseen batch? Two variants: `bacc` keeps DMSO as
    # a ninth class (balanced: DMSO is ~76% of wells), `bacc_xdmso` drops DMSO from
    # both sides (eight-way, chance 1/8, comparable with the integration probe of
    # probe_integration.py). The thesis reports `bacc_xdmso`.
    bacc = M.probe_balanced_accuracy(Z_Rw, meta_Rw, Z_Hw, meta_Hw, seed=SEED)
    bacc_xdmso = M.probe_balanced_accuracy(Z_Rw, meta_Rw, Z_Hw, meta_Hw,
                                           control_key="DMSO", seed=SEED)

    rng = np.random.default_rng(SEED)
    ref = np.arange(Z_Rw.shape[0])
    if len(ref) > N_REF_SUB:
        ref = rng.choice(ref, size=N_REF_SUB, replace=False)
    Zc = np.vstack([Z_Rw[ref], Z_Hw])
    mc = pd.concat([meta_Rw.iloc[ref], meta_Hw], ignore_index=True).copy()
    mc["holdout"] = np.r_[np.zeros(len(ref), int), np.ones(len(meta_Hw), int)].astype(str)
    sil = M.compute_silhouette(Zc, mc, ["holdout"])["silhouette_holdout"]
    ili = M.compute_lisi(Zc, mc, ["holdout"])["lisi_holdout"]
    kb = M.compute_kbet(Zc, mc, "holdout")["kbet_holdout_rejection"]
    return dict(auroc=auroc, auroc_xref=auroc_xref, bacc=bacc, bacc_xdmso=bacc_xdmso,
                sil=sil, ilisi=ili, kbet=kb, n_wells=len(meta_Hw))


def boot_ci(deltas, B=10000, seed=0):
    d = np.asarray(deltas, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="*", default=None,
                    help="subset of batch codes to use as held-out folds (default: all present)")
    ap.add_argument("--dmso-subsample", type=int, default=None,
                    help="cap DMSO wells/batch in the TVN fit (speed; default None = canonical)")
    ap.add_argument("--methods", nargs="*", default=["raw", "tvn"])
    ap.add_argument("--max-sites-per-batch", type=int, default=None,
                    help="cap sites/batch when POOLING (login-node logic validation only; "
                         "NOT for real results — shrinks every batch)")
    ap.add_argument("--ce", type=Path, default=CE,
                    help="channelwise_embeddings dir of the run (default: canonical ResNet; "
                         "for MAE pass checkpoints/honest-snowflake-408_x79nwnqn/channelwise_embeddings)")
    ap.add_argument("--meta-path", type=Path, default=META_PATH,
                    help="metadata parquet (default: source_3_w_neg.pq; for Source-1 pass source_1_w_neg.pq)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pool ALL data; LOBO defines its own folds by batch ---
    src = pd.read_parquet(args.meta_path)
    parts = []
    for split in ("train", "validation", "test"):
        X, m = load_split(args.ce / f"{split}_channelwise_embeddings.npz", src, "DMSO")
        parts.append((X, m))
    X_all = np.vstack([p[0] for p in parts])
    meta_all = pd.concat([p[1] for p in parts], ignore_index=True)
    batches = meta_all[BATCH_KEY].astype(int).to_numpy()
    if args.max_sites_per_batch:
        rng = np.random.default_rng(SEED)
        keep = []
        for b in np.unique(batches):
            idx = np.where(batches == b)[0]
            if len(idx) > args.max_sites_per_batch:
                idx = rng.choice(idx, size=args.max_sites_per_batch, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        X_all, meta_all = X_all[keep], meta_all.iloc[keep].reset_index(drop=True)
        batches = meta_all[BATCH_KEY].astype(int).to_numpy()
        print(f"[subsample] capped to {args.max_sites_per_batch}/batch -> X {X_all.shape} "
              "(LOGIC VALIDATION ONLY — numbers not meaningful)", flush=True)
    folds = sorted(np.unique(batches)) if args.batches is None else args.batches
    print(f"[pool] X {X_all.shape}  batches={sorted(np.unique(batches))}  folds={folds}", flush=True)

    rows = []
    for b in folds:
        t0 = time.time()
        H = batches == b
        X_R, meta_R = X_all[~H], meta_all.loc[~H].reset_index(drop=True)
        X_H, meta_H = X_all[H], meta_all.loc[H].reset_index(drop=True)
        for name in args.methods:
            Z_R, Z_H = CORRECTORS[name](X_R, meta_R, X_H, meta_H, BATCH_KEY,
                                        dmso_subsample=args.dmso_subsample)
            s = score_holdout(Z_R, meta_R, Z_H, meta_H)
            rows.append(dict(batch=int(b), method=name, **s))
            del Z_R, Z_H
        r = {x["method"]: x for x in rows if x["batch"] == b}
        d = {k: r["tvn"][k] - r["raw"][k]
             for k in ("auroc", "auroc_xref", "bacc", "sil", "ilisi", "kbet")} if "tvn" in r else {}
        print(f"[fold b={b:2d}] n={r['raw']['n_wells']:4d}  "
              f"raw(AUx {r['raw']['auroc_xref']:.3f} BA {r['raw']['bacc']:.3f} iL {r['raw']['ilisi']:.2f} kB {r['raw']['kbet']:.2f})  "
              + (f"tvn(AUx {r['tvn']['auroc_xref']:.3f} BA {r['tvn']['bacc']:.3f} iL {r['tvn']['ilisi']:.2f} kB {r['tvn']['kbet']:.2f})  "
                 f"Δ[AUx {d['auroc_xref']:+.3f} BA {d['bacc']:+.3f} iL {d['ilisi']:+.2f} kB {d['kbet']:+.2f}]" if d else "")
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        del X_R, X_H

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "tvn_lobo_per_batch.csv", index=False)

    # --- paired deltas (tvn - raw) per fold, mean + bootstrap CI over batches ---
    piv = df.pivot(index="batch", columns="method")
    run_name = args.ce.resolve().parents[0].name  # checkpoints/<run>/channelwise_embeddings -> <run>
    md = (f"# TVN leave-one-batch-out ({run_name}, split=batches)\n\n"
          f"_n={piv.shape[0]} held-out batches; each fold fits on the other {piv.shape[0]-1}, transforms the "
          f"held-out batch via its own DMSO, scores the held-out batch only. Well-aggregated. "
          f"dmso_subsample={args.dmso_subsample}. Mean Δ (tvn − raw) [95% bootstrap CI over batches]. "
          "tvn_lobo.py_\n\n")
    read = {"auroc_xref": "bio preserved, H-vs-reference, DMSO-excluded (want ≥0)",
            "auroc": "legacy: within-H, DMSO-in (want ≥0)",
            "bacc": "bio: compounds classifiable, 9-way DMSO-in (want ≥0)",
            "bacc_xdmso": "bio: compounds classifiable, 8-way DMSO-excluded (want ≥0)",
            "ilisi": "mixing (want +)",
            "sil": "mixing (want −)", "kbet": "mixing (want −)"}
    for method in [m for m in args.methods if m != "raw"]:
        md += f"\n### {method} − raw  (mean Δ [95% bootstrap CI over batches])\n\n"
        md += "| metric | mean Δ | 95% CI | reading |\n|---|---|---|---|\n"
        for k in ("auroc_xref", "auroc", "bacc", "bacc_xdmso", "ilisi", "sil", "kbet"):
            dvals = (piv[k][method] - piv[k]["raw"]).to_numpy()
            mean, lo, hi = boot_ci(dvals)
            md += f"| Δ {k} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {read[k]} |\n"
    md += ("\n_Primary endpoints: Δ auroc_xref (bio) and Δ iLISI (mixing). 'Generalizes' = "
           "iLISI CI strictly above 0 AND AUROC CI not meaningfully below 0. "
           "Compare methods: best generalizer = largest Δ iLISI with Δ AUROC CI not below 0._\n")
    (args.out_dir / "tvn_lobo_summary.md").write_text(md)
    print("\n" + md)
    print(f"[wrote] {args.out_dir/'tvn_lobo_per_batch.csv'}\n[wrote] {args.out_dir/'tvn_lobo_summary.md'}")


if __name__ == "__main__":
    main()
