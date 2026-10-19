"""
common.py — Shared utilities for normalize_*.py scripts.

Provides the boilerplate that every batch-correction script needs:
  - STRATEGY_TO_BATCH_KEYS  mapping from split strategy to batch columns
  - detect_batch_keys()     auto-detect batch keys from split_keys.json
  - load_split()            load a channelwise .npz and align to metadata
  - scale_and_pca()         StandardScaler + PCA fitted on train, applied to all splits
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


STRATEGY_TO_BATCH_KEYS = {
    "batches": ["batch"],
    "plates":  ["plate"],
    "wells":   ["well"],
    # The random split has no natural held-out technical unit. Score and correct
    # against batch only, so it is a clean "same target (batch), random training
    # partition" control on the batches split. Do not add plate/well here:
    # harmony.py corrects over the whole key list and would over-correct.
    "random":  ["batch"],
}


def detect_batch_keys(input_dir: Path):
    """
    Read split_keys.json one level above input_dir and return the
    corresponding batch key list, or None if the file is absent.
    """
    split_keys_path = input_dir.parent / "split_keys.json"
    if not split_keys_path.exists():
        return None
    with open(split_keys_path) as f:
        info = json.load(f)
    strategy = info.get("split_strategy")
    keys = STRATEGY_TO_BATCH_KEYS.get(strategy)
    if keys:
        print(f"Auto-detected split_strategy={strategy!r} -> batch_keys={keys}")
    return keys


def load_split(npz_path, source_3: pd.DataFrame, control_key: str):
    """
    Load a channelwise .npz embedding file and align rows to the metadata
    parquet by Metadata_Sample_ID.

    Returns
    -------
    X    : np.ndarray  [N, D]  raw embeddings
    meta : pd.DataFrame [N]   aligned metadata with standardised column names
    """
    npz = np.load(npz_path, allow_pickle=True)
    X = npz["emb"]
    names = npz["names"].astype(str)

    meta_idx = source_3.set_index("Metadata_Sample_ID", drop=False)
    aligned = meta_idx.reindex(names)
    missing = aligned["Metadata_Sample_ID"].isna().sum()
    if missing:
        raise ValueError(f"{missing} names in {npz_path} not found in metadata")

    aligned["is_control"] = aligned["pert_iname_str"].eq(control_key)
    aligned = aligned.rename(columns={
        "Metadata_Plate": "plate",
        "Metadata_Well":  "well",
        "Metadata_Batch": "batch",
        "pert_iname":     "label_id",
        "pert_iname_str": "label_name",
    })
    return X, aligned.reset_index(drop=True)


def scale_and_pca(X_tr, X_va, X_te, n_components: int, seed: int = 42):
    """
    Fit StandardScaler and PCA on X_tr, then transform X_va and X_te.

    Parameters
    ----------
    X_tr, X_va, X_te : np.ndarray  raw embeddings for each split
    n_components      : int         desired PCA dimensionality (capped automatically)
    seed              : int         random seed for PCA

    Returns
    -------
    Z_tr, Z_va, Z_te : np.ndarray  PCA-projected embeddings
    n_comp           : int          actual number of components used
    ev               : float        cumulative explained variance ratio
    pca              : PCA          fitted PCA object (needed for inverse_transform)
    """
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    n_comp = min(n_components, X_tr_sc.shape[1], X_tr_sc.shape[0])
    pca = PCA(n_components=n_comp, random_state=seed)
    Z_tr = pca.fit_transform(X_tr_sc)
    Z_va = pca.transform(X_va_sc)
    Z_te = pca.transform(X_te_sc)
    ev = float(pca.explained_variance_ratio_.sum())

    return Z_tr, Z_va, Z_te, n_comp, ev, pca
