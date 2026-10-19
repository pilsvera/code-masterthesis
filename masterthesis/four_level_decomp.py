#!/usr/bin/env python
"""Four-level nested variance decomposition: source / batch / plate / well.

DMSO wells of Source 3 (canonical run, train+val) and Source 1 (all splits of the
xsrc dir, S1 images encoded through the S3 weights), in ONE shared 50-PC space
fitted on the Source-3 train split -- the same convention as the cross-source
transfer. Plus per-level probes.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import sys
from pathlib import Path
import numpy as np, pandas as pd

# Repository root (the directory that holds masterthesis/ and checkpoints/);
# override with THESIS_REPO=/path if the checkpoints live elsewhere.
REPO = Path(os.environ.get("THESIS_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO / "masterthesis"))
from common import load_split, scale_and_pca
from eval_all import aggregate_to_well
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CONFIGS = {
    "resnet": {"s3": REPO/"checkpoints/stoic-snowball-398_9llyzmze/channelwise_embeddings",
               "s1": REPO/"checkpoints/xsrc_resnet_s3on_s1/channelwise_embeddings"},
    "mae":    {"s3": REPO/"checkpoints/honest-snowflake-408_x79nwnqn/channelwise_embeddings",
               "s1": REPO/"checkpoints/xsrc_mae_s3on_s1/channelwise_embeddings"},
}

def load_pool(ce, meta, splits):
    Xs, Ms = [], []
    for sp in splits:
        f = ce / f"{sp}_channelwise_embeddings.npz"
        if not f.exists(): continue
        X, m = load_split(f, meta, "DMSO")
        Xs.append(X); Ms.append(m)
    return np.vstack(Xs), pd.concat(Ms, ignore_index=True)

meta3 = pd.read_parquet(REPO/"source_3_w_neg.pq")
meta1 = pd.read_parquet(REPO/"source_1_w_neg.pq")

for enc, cfg in CONFIGS.items():
    print(f"\n=== {enc} ===", flush=True)
    # shared space: scaler+PCA fit on S3 train only (transfer convention)
    X3tr, m3tr = load_pool(cfg["s3"], meta3, ["train"])
    X3va, m3va = load_pool(cfg["s3"], meta3, ["validation"])
    X1,   m1   = load_pool(cfg["s1"], meta1, ["train","validation","test"])
    Z3tr, Z3va, Z1, _, _, _ = scale_and_pca(X3tr, X3va, X1, 50)
    Z = np.vstack([Z3tr, Z3va, Z1])
    m3 = pd.concat([m3tr, m3va], ignore_index=True); m3["source"]="Source 3"
    m1 = m1.copy(); m1["source"]="Source 1"
    meta = pd.concat([m3, m1], ignore_index=True)
    # disambiguate batch/plate names across sources
    meta["batch"] = meta["source"].str[-1] + "|" + meta["batch"].astype(str)
    meta["plate"] = meta["batch"] + "|" + meta["plate"].astype(str)

    Zw, mw = aggregate_to_well(Z, meta)
    dm = mw["is_control"].values
    Zd, md = Zw[dm], mw[dm].reset_index(drop=True)
    print(f"DMSO wells: {len(md)} (S3 {sum(md['source']=='Source 3')}, "
          f"S1 {sum(md['source']=='Source 1')}); batches {md['batch'].nunique()}, "
          f"plates {md['plate'].nunique()}")

    # ---- nested decomposition: source / batch(source) / plate(batch) / residual ----
    g = Zd - Zd.mean(0); tot=(g**2).sum()
    sm = np.zeros_like(g)
    for _,idx in md.groupby("source").indices.items(): sm[idx]=g[idx].mean(0)
    bm = np.zeros_like(g)
    for _,idx in md.groupby("batch").indices.items(): bm[idx]=g[idx].mean(0)
    pm = np.zeros_like(g)
    for _,idx in md.groupby("plate").indices.items(): pm[idx]=g[idx].mean(0)
    print("variance fractions:")
    print(f"  source          : {(sm**2).sum()/tot:.3f}")
    print(f"  batch-in-source : {((bm-sm)**2).sum()/tot:.3f}")
    print(f"  plate-in-batch  : {((pm-bm)**2).sum()/tot:.3f}")
    print(f"  residual (well) : {((g-pm)**2).sum()/tot:.3f}")

    # ---- probes ----
    def probe(y, groups=None):
        clf=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        yp=cross_val_predict(clf, Zd if groups is None else Zd[groups], y,
                             cv=StratifiedKFold(5,shuffle=True,random_state=0))
        n=len(set(y)); return balanced_accuracy_score(y,yp)*n
    print(f"probes (x chance): source {probe(md['source'].values):.1f}", flush=True)
    print(f"                   batch  {probe(md['batch'].values):.1f}")
    xs=[]
    for b,sub in md.groupby("batch"):
        y=sub["plate"].values
        if len(set(y))<2: continue
        xs.append(probe(y, sub.index.values))
    print(f"                   plate  {np.mean(xs):.1f} (mean over batches, range {min(xs):.1f}-{max(xs):.1f})")
