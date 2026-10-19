"""
fastmnn.py — fastMNN batch correction (pure Python, no mnnpy dependency).

Called by normalize_pipeline.py. Not a standalone script.

Implements the core MNN correction algorithm:
1. For each pair of batches, find mutual nearest neighbours (MNN pairs)
2. Compute correction vectors from MNN pairs
3. Extend correction to all cells via Gaussian-weighted average of MNN vectors

This is distinct from Scanorama (mnn.py) which uses a different MNN-based
algorithm with panoramic stitching.

Reference: Haghverdi et al. 2018 Nature Biotechnology,
"Batch effects in single-cell RNA-sequencing data are corrected by matching
mutual nearest neighbors".
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from common import scale_and_pca


def _find_mnn_pairs(X_ref, X_query, k=20):
    """Find mutual nearest neighbour pairs between two datasets."""
    nn_ref = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="euclidean")
    nn_ref.fit(X_ref)
    nn_query = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="euclidean")
    nn_query.fit(X_query)

    # For each query point, find k nearest in ref
    _, idx_q2r = nn_ref.kneighbors(X_query)
    # For each ref point, find k nearest in query
    _, idx_r2q = nn_query.kneighbors(X_ref)

    # Find mutual pairs
    pairs = []
    for q_idx in range(X_query.shape[0]):
        for r_idx in idx_q2r[q_idx]:
            if q_idx in idx_r2q[r_idx]:
                pairs.append((r_idx, q_idx))

    return pairs


def _correct_batch(X_ref, X_query, k=20, sigma=1.0, n_anchors=20):
    """
    Correct X_query toward X_ref using MNN pairs.

    Uses a k-NN index on MNN anchor points so each query cell only looks up
    its nearest `n_anchors` MNN pairs instead of computing distances to all
    pairs (which causes memory corruption on large batches).

    Returns corrected X_query.
    """
    pairs = _find_mnn_pairs(X_ref, X_query, k=k)

    if len(pairs) == 0:
        print(f"    WARNING: No MNN pairs found (k={k}). Returning uncorrected.")
        return X_query.copy()

    ref_idx = np.array([p[0] for p in pairs])
    query_idx = np.array([p[1] for p in pairs])

    # Correction vectors at MNN pair locations
    correction_vectors = X_ref[ref_idx] - X_query[query_idx]
    mnn_query_locs = X_query[query_idx]

    print(f"    Found {len(pairs)} MNN pairs, using {min(n_anchors, len(pairs))} nearest anchors per cell")

    # Build k-NN index on MNN anchor locations
    n_use = min(n_anchors, len(pairs))
    nn = NearestNeighbors(n_neighbors=n_use, algorithm="ball_tree", metric="euclidean")
    nn.fit(mnn_query_locs)

    # Batch query: for each cell, find its nearest MNN anchors
    dists, indices = nn.kneighbors(X_query)  # (n_query, n_use)

    # Gaussian weights from distances
    weights = np.exp(-dists**2 / (2 * sigma**2))  # (n_query, n_use)
    w_sum = weights.sum(axis=1, keepdims=True)     # (n_query, 1)
    w_sum = np.maximum(w_sum, 1e-10)

    # Weighted average of correction vectors for each query cell
    # correction_vectors[indices] -> (n_query, n_use, n_dim)
    local_corrections = correction_vectors[indices]
    correction = (weights[:, :, None] * local_corrections).sum(axis=1) / w_sum

    X_corrected = X_query + correction
    return X_corrected


def apply(X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
          batch_key, n_components, k=20, sigma=1.0):
    """
    StandardScaler + PCA (fit on train only), then fastMNN on train+val.
    Test receives uncorrected PCA embeddings (no transform API).

    Batches are corrected sequentially: the first (largest) batch is used as
    reference, and each subsequent batch is corrected toward the growing
    reference set.

    Returns (Z_trainval_corr, meta_trainval, Z_te, meta_te, pca).
    """
    Z_tr, Z_va, Z_te, n_comp, ev, pca = scale_and_pca(
        X_tr, X_va, X_te, n_components)
    print(f"  [fastmnn] PCA: {X_tr.shape[1]}D -> {n_comp}D  "
          f"(explained variance: {ev:.3f})")

    Z_trainval = np.vstack([Z_tr, Z_va])
    meta_trainval = pd.concat([meta_tr, meta_va], ignore_index=True)

    batch_labels = meta_trainval[batch_key].astype(str).values
    unique_batches = list(dict.fromkeys(batch_labels))

    # Sort batches by size (largest first = reference)
    batch_sizes = {b: (batch_labels == b).sum() for b in unique_batches}
    unique_batches = sorted(unique_batches, key=lambda b: -batch_sizes[b])

    print(f"  [fastmnn] {len(unique_batches)} batches, "
          f"sizes: {[batch_sizes[b] for b in unique_batches]} | "
          f"k={k} | sigma={sigma}")

    # Split by batch, preserving original indices
    batch_data = {}
    for b in unique_batches:
        idx = np.where(batch_labels == b)[0]
        batch_data[b] = {"idx": idx, "Z": Z_trainval[idx]}

    # Sequential correction: first batch is reference
    Z_corrected = np.empty_like(Z_trainval)
    ref_batch = unique_batches[0]
    Z_corrected[batch_data[ref_batch]["idx"]] = batch_data[ref_batch]["Z"]

    corrected_idx = list(batch_data[ref_batch]["idx"])
    ref_Z = batch_data[ref_batch]["Z"].copy()

    for i, b in enumerate(unique_batches[1:], 1):
        print(f"  [fastmnn] Correcting batch {i}/{len(unique_batches)-1}: "
              f"'{b}' ({batch_sizes[b]} samples) -> reference ({len(corrected_idx)} samples)")
        corrected = _correct_batch(ref_Z, batch_data[b]["Z"], k=k, sigma=sigma)
        Z_corrected[batch_data[b]["idx"]] = corrected

        # Grow reference
        ref_Z = np.vstack([ref_Z, corrected])
        corrected_idx.extend(batch_data[b]["idx"])

    Z_trainval_corr = Z_corrected.astype(np.float32)
    print(f"  [fastmnn] Output: {Z_trainval_corr.shape}")

    return Z_trainval_corr, meta_trainval, Z_te, meta_te, pca
