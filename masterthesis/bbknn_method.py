"""
bbknn_method.py — BBKNN (Batch Balanced KNN) batch correction.

Called by normalize_pipeline.py. Not a standalone script.

BBKNN corrects the KNN graph to be batch-balanced, then embeddings are
extracted via UMAP on the corrected graph. This means results are in
UMAP space, not PCA space — not directly comparable to PCA-space methods.

Reference: Polanski et al. 2020 Bioinformatics,
"BBKNN: fast batch alignment of single cell transcriptomes".
"""

import numpy as np
import pandas as pd

from common import scale_and_pca


MAX_BATCHES_DEFAULT = 100


class BBKNNIntractableError(RuntimeError):
    """Raised when BBKNN would be too expensive to run on this batch_key.

    BBKNN's neighbor computation scales roughly as O(num_batches^2). With
    batch_key='well' (thousands of wells on JUMP-CP Source 3) the call
    runs for >30 h without progress and produces no output. Surfaced
    as a typed error so the caller can mark
    the method 'SKIPPED' rather than hang the whole array.
    """


def apply(X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
          batch_key, n_components, neighbors_within_batch=25,
          max_batches=MAX_BATCHES_DEFAULT):
    """
    StandardScaler + PCA (fit on train only), then BBKNN on train+val.
    Corrected embeddings are extracted via UMAP on the batch-balanced graph.
    Test receives uncorrected PCA embeddings (no transform API).

    Raises BBKNNIntractableError if the number of unique values in
    batch_key exceeds max_batches (default 100). BBKNN cost is
    ~O(num_batches^2); for split_strategy='wells' on JUMP-CP this is
    thousands of batches and the call never finishes.

    Returns (Z_trainval_corr, meta_trainval, Z_te, meta_te, pca).
    """
    import bbknn
    import scanpy as sc
    import anndata as ad

    Z_tr, Z_va, Z_te, n_comp, ev, pca = scale_and_pca(
        X_tr, X_va, X_te, n_components)
    print(f"  [bbknn] PCA: {X_tr.shape[1]}D -> {n_comp}D  "
          f"(explained variance: {ev:.3f})")

    Z_trainval = np.vstack([Z_tr, Z_va])
    meta_trainval = pd.concat([meta_tr, meta_va], ignore_index=True)

    n_batches = meta_trainval[batch_key].nunique()
    if n_batches > max_batches:
        raise BBKNNIntractableError(
            f"BBKNN: batch_key='{batch_key}' has {n_batches} unique values "
            f"(> max_batches={max_batches}). Cost is O(num_batches^2) and "
            f"BBKNN hangs indefinitely above ~100 batches on JUMP-CP. "
            f"Skipping. Override with --bbknn-max-batches=<N> if you have "
            f"the compute budget to wait.")

    adata = ad.AnnData(Z_trainval)
    adata.obs[batch_key] = meta_trainval[batch_key].astype(str).values
    adata.obsm["X_pca"] = Z_trainval

    print(f"  [bbknn] Running BBKNN on train+val ({Z_trainval.shape[0]} samples) "
          f"batch_key='{batch_key}' | n_batches={n_batches} | "
          f"neighbors_within_batch={neighbors_within_batch} ...")

    bbknn.bbknn(adata, batch_key=batch_key, use_rep="X_pca",
                neighbors_within_batch=neighbors_within_batch)

    # Extract embedding via UMAP on the corrected graph
    sc.tl.umap(adata, n_components=n_comp)
    Z_trainval_corr = adata.obsm["X_umap"].astype(np.float32)

    print(f"  [bbknn] Output: {Z_trainval_corr.shape} (UMAP on corrected graph)")

    return Z_trainval_corr, meta_trainval, Z_te, meta_te, pca
