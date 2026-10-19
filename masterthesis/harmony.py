"""
harmony.py — Harmony batch correction.

Called by normalize_pipeline.py. Not a standalone script.
"""

import numpy as np
import pandas as pd
import harmonypy as hm

from common import scale_and_pca


def apply(X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
          batch_keys, n_components):
    """
    StandardScaler + PCA (fit on train only), then Harmony on train+val jointly.
    Test receives PCA projection only (no Harmony; test is unused in the benchmark).

    Harmony is a joint embedder with no out-of-sample transform, and the benchmark
    evaluates train+val combined, so Harmony is run on the full train+val PCA
    matrix at once — consistent with every other method (ComBat/scVI/MNN/fastMNN/
    BBKNN correct train+val jointly; TVN transforms val per-batch). Only the batch
    keys are used (never the compound labels), so this is consistency, not leakage.

    (Previously Harmony was run on train ONLY and the *uncorrected* PCA-only val
    was stacked on top, leaving ~20% of the eval pool — whole held-out batches for
    the batches split — in raw space and making Harmony's batch-mixing metrics
    non-comparable to the other methods.)

    Returns (Z_trainval, meta_trainval, Z_te, meta_te, pca).
    """
    Z_tr, Z_va, Z_te, n_comp, ev, pca = scale_and_pca(
        X_tr, X_va, X_te, n_components)
    print(f"  [harmony] PCA: {X_tr.shape[1]}D -> {n_comp}D  (explained variance: {ev:.3f})")

    Z_trainval_pca = np.vstack([Z_tr, Z_va])
    meta_trainval  = pd.concat([meta_tr, meta_va], ignore_index=True)

    meta_h = meta_trainval.reset_index(drop=True).copy()
    for key in batch_keys:
        meta_h[key] = meta_h[key].astype(str)
    print(f"  [harmony] Running Harmony on train+val ({len(meta_h)} samples) "
          f"correcting for: {batch_keys} ...")
    ho = hm.run_harmony(Z_trainval_pca, meta_h, batch_keys)
    Z_trainval = np.asarray(ho.Z_corr)
    if Z_trainval.shape[0] == n_comp and Z_trainval.shape[1] != n_comp:
        Z_trainval = Z_trainval.T   # numpy backend: (n_comp, N) → (N, n_comp)

    return Z_trainval, meta_trainval, Z_te, meta_te, pca
