"""
desc_method.py — DESC (Deep Embedding for Single-cell Clustering) batch correction.

Called by normalize_pipeline.py. Not a standalone script.

DESC uses a deep autoencoder with iterative Louvain clustering to learn
a batch-corrected low-dimensional embedding. Output dimensionality is
determined by hidden_dims (default 32D), not n_components.

Requires: pip install git+https://github.com/johnarevalo/desc.git
          pip install tensorflow-cpu  (or tensorflow)

Reference: Li et al. 2020 Nature Communications,
"Deep learning enables accurate clustering with batch effect removal in
single-cell RNA-seq analysis".
"""

import numpy as np
import pandas as pd

from common import scale_and_pca


def apply(X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
          batch_key, n_components,
          hidden_dims=None, louvain_resolution=1.0,
          n_neighbors=64, batch_size=1024, tol=0.05,
          use_gpu=False, gpu_id=0):
    """
    StandardScaler + PCA (fit on train only), then DESC on train+val.
    Test receives uncorrected PCA embeddings (no transform API).

    NOTE: DESC produces a fixed-dimensional embedding (default 32D via
    hidden_dims=[n_comp, 128, 32]), not n_components dimensions.

    Returns (Z_trainval_corr, meta_trainval, Z_te, meta_te, pca).
    """
    import tempfile
    import anndata as ad

    try:
        from desc import train as desc_train, scale_bygroup
    except ImportError:
        raise ImportError(
            "DESC not installed. Install with:\n"
            "  pip install git+https://github.com/johnarevalo/desc.git\n"
            "  pip install tensorflow-cpu"
        )

    Z_tr, Z_va, Z_te, n_comp, ev, pca = scale_and_pca(
        X_tr, X_va, X_te, n_components)
    print(f"  [desc] PCA: {X_tr.shape[1]}D -> {n_comp}D  "
          f"(explained variance: {ev:.3f})")

    Z_trainval = np.vstack([Z_tr, Z_va]).astype(np.float32)
    meta_trainval = pd.concat([meta_tr, meta_va], ignore_index=True)

    adata = ad.AnnData(X=Z_trainval)
    adata.obs[batch_key] = meta_trainval[batch_key].astype(str).values

    if hidden_dims is None:
        hidden_dims = [n_comp, 128, 32]

    print(f"  [desc] Training DESC autoencoder: dims={hidden_dims} | "
          f"resolution={louvain_resolution} | tol={tol} | "
          f"n_neighbors={n_neighbors} | batch_size={batch_size}")

    scale_bygroup(adata, batch_key, max_value=None)

    save_dir = tempfile.mkdtemp()
    adata = desc_train(
        adata, dims=hidden_dims, tol=tol,
        n_neighbors=n_neighbors, batch_size=batch_size,
        louvain_resolution=louvain_resolution,
        save_encoder_weights=False,
        save_dir=save_dir,
        do_tsne=False, use_GPU=use_gpu, GPU_id=gpu_id,
        use_ae_weights=False, do_umap=False,
    )

    embed_key = f"X_Embeded_z{louvain_resolution}"
    Z_trainval_corr = adata.obsm[embed_key].astype(np.float32)

    print(f"  [desc] Output: {Z_trainval_corr.shape} "
          f"(from '{embed_key}', {hidden_dims[-1]}D embedding)")

    return Z_trainval_corr, meta_trainval, Z_te, meta_te, pca
