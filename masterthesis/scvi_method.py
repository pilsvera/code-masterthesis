"""
scvi_method.py — scVI batch correction.

Named scvi_method.py (not scvi.py) to avoid shadowing the scvi package.
Called by normalize_pipeline.py. Not a standalone script.
"""

import numpy as np
import pandas as pd
import torch
import scvi
import anndata as ad

from common import scale_and_pca

torch.set_float32_matmul_precision("high")


def apply(X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
          batch_key, n_components,
          n_latent=50, n_hidden=256, n_layers=2,
          max_epochs=200, lr=1e-3, seed=42):
    """
    StandardScaler + PCA (fit on train only), then scVI on train+val combined.
    Test receives uncorrected PCA embeddings (scVI has no transform API for
    held-out data).

    Returns (Z_trainval, meta_trainval, Z_te, meta_te, pca).
    Z_trainval shape: [N_tr+N_va, n_latent]
    Z_te shape:       [N_te, n_comp]  (PCA-only)
    """
    scvi.settings.seed = seed

    Z_tr, Z_va, Z_te, n_comp, ev, pca = scale_and_pca(
        X_tr, X_va, X_te, n_components, seed=seed)
    print(f"  [scvi] PCA: {X_tr.shape[1]}D -> {n_comp}D  (explained variance: {ev:.3f})")

    Z_trainval    = np.vstack([Z_tr, Z_va]).astype(np.float32)
    meta_trainval = pd.concat([meta_tr, meta_va], ignore_index=True)

    adata = ad.AnnData(X=Z_trainval)
    adata.obs[batch_key] = meta_trainval[batch_key].astype(str).values

    print(f"  [scvi] Running scVI on train+val ({Z_trainval.shape[0]} samples) "
          f"correcting for: '{batch_key}' ...")
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi.model.SCVI(
        adata,
        gene_likelihood="normal",
        n_latent=min(n_latent, n_comp),
        n_hidden=n_hidden,
        n_layers=n_layers,
        log_variational=False,
        use_observed_lib_size=False,
    )
    model.train(
        max_epochs=max_epochs,
        batch_size=256,
        plan_kwargs={"lr": lr},
        early_stopping=True,
        early_stopping_patience=20,
        gradient_clip_val=1.0,
    )

    Z_trainval_corr = model.get_latent_representation()
    print(f"  [scvi] Latent shape: {Z_trainval_corr.shape}")

    return Z_trainval_corr, meta_trainval, Z_te, meta_te, pca
