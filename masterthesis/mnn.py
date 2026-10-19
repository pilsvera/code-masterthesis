"""
mnn.py — Scanorama (MNN-based) batch correction.

Called by normalize_pipeline.py. Not a standalone script.

Reference: Hie et al. 2019 Nature Biotechnology, "Efficient integration of
heterogeneous single-cell transcriptomes using Scanorama".
"""

import numpy as np
import pandas as pd
import scanorama
from sklearn.neighbors import NearestNeighbors

from common import scale_and_pca


def _run_scanorama(Z: np.ndarray, batch: pd.Series,
                   k: int, sigma: float, alpha: float, approx: bool,
                   geosketch: bool, geosketch_max: int) -> np.ndarray:
    """Apply Scanorama correction to Z [N, D]. Returns Z_corrected [N, D]."""
    batch_labels   = batch.astype(str).values
    unique_batches = list(dict.fromkeys(batch_labels))

    batch_arrays  = []
    batch_indices = []
    for b in unique_batches:
        idx = np.where(batch_labels == b)[0]
        batch_arrays.append(Z[idx].astype(np.float32))
        batch_indices.append(idx)

    print(f"  [mnn] Scanorama: {len(unique_batches)} batches, sizes: "
          f"{[len(i) for i in batch_indices]}")

    feature_names = [f"PC{i}" for i in range(Z.shape[1])]
    genes_list    = [feature_names] * len(batch_arrays)

    if geosketch and any(len(a) > geosketch_max for a in batch_arrays):
        rng = np.random.default_rng(42)
        sketch_arrays    = []
        sketch_local_idx = []
        for arr in batch_arrays:
            if len(arr) > geosketch_max:
                lidx = rng.choice(len(arr), size=geosketch_max, replace=False)
                lidx.sort()
            else:
                lidx = np.arange(len(arr))
            sketch_arrays.append(arr[lidx])
            sketch_local_idx.append(lidx)

        corrected_sketches, _ = scanorama.correct(
            sketch_arrays, genes_list,
            return_dimred=False, knn=k, sigma=sigma, alpha=alpha, approx=approx,
        )

        Z_corr = np.empty_like(Z, dtype=np.float32)
        for global_idx, arr, lidx, corr in zip(
                batch_indices, batch_arrays, sketch_local_idx, corrected_sketches):
            if len(lidx) == len(arr):
                Z_corr[global_idx] = corr
            else:
                sketch = arr[lidx]
                delta  = corr - sketch
                nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(sketch)
                _, neighbors = nn.kneighbors(arr)
                Z_corr[global_idx] = arr + delta[neighbors[:, 0]]
    else:
        corrected_list, _ = scanorama.correct(
            batch_arrays, genes_list,
            return_dimred=False, knn=k, sigma=sigma, alpha=alpha, approx=approx,
        )
        Z_corr = np.empty_like(Z, dtype=np.float32)
        for idx, corr in zip(batch_indices, corrected_list):
            # Scanorama can return scipy.sparse.csr_matrix instead of ndarray
            # when input dtypes vary across batches; densify before assignment.
            if hasattr(corr, "toarray"):
                corr = corr.toarray()
            Z_corr[idx] = np.asarray(corr, dtype=np.float32)

    return Z_corr


def apply(X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
          batch_key, n_components,
          k=20, sigma=15.0, alpha=0.1, approx=True,
          geosketch=True, geosketch_max=5000):
    """
    StandardScaler + PCA (fit on train only), then Scanorama on train+val combined.
    Test receives uncorrected PCA embeddings (Scanorama has no transform API).

    Returns (Z_trainval, meta_trainval, Z_te, meta_te, pca).
    """
    Z_tr, Z_va, Z_te, n_comp, ev, pca = scale_and_pca(
        X_tr, X_va, X_te, n_components)
    print(f"  [mnn] PCA: {X_tr.shape[1]}D -> {n_comp}D  (explained variance: {ev:.3f})")

    Z_trainval    = np.vstack([Z_tr, Z_va])
    meta_trainval = pd.concat([meta_tr, meta_va], ignore_index=True)

    sketch_info = f"geosketch_max={geosketch_max}" if geosketch else "no geosketch"
    print(f"  [mnn] Running Scanorama on train+val ({Z_trainval.shape[0]} samples) "
          f"batch_key='{batch_key}' | k={k} | sigma={sigma} | {sketch_info} ...")

    Z_trainval_corr = _run_scanorama(
        Z_trainval, meta_trainval[batch_key],
        k, sigma, alpha, approx, geosketch, geosketch_max,
    )

    return Z_trainval_corr, meta_trainval, Z_te, meta_te, pca
