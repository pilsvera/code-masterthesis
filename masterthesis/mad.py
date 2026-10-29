"""
mad.py — Per-plate MAD normalization.

Called by normalize_pipeline.py. Not a standalone script.
"""

import numpy as np
import pandas as pd


def normalize(X: np.ndarray, meta: pd.DataFrame,
              batch_key: str, eps: float = 1e-8) -> np.ndarray:
    """
    Per-plate MAD normalization using DMSO (negative control) wells.

    For each plate p and embedding dimension d:
        Z[p, d] = (X[p, d] - median_DMSO[p, d]) / (MAD_DMSO[p, d] + eps)

    Stats are computed from that plate's own DMSO wells, so each split
    can be normalised independently with no cross-split leakage.
    Plates with no DMSO wells are left unchanged (a warning is printed).
    """
    X_out = X.astype(float, copy=True)

    for plate in meta[batch_key].unique():
        plate_mask = (meta[batch_key] == plate).values
        dmso_mask  = plate_mask & meta["is_control"].values

        if dmso_mask.sum() == 0:
            print(f"  WARNING: no DMSO wells for {batch_key}={plate!r} — "
                  f"leaving {plate_mask.sum()} rows unnormalised.")
            continue

        X_neg   = X[dmso_mask]
        medians = np.median(X_neg, axis=0)
        mads    = np.median(np.abs(X_neg - medians), axis=0)
        X_out[plate_mask] = (X[plate_mask] - medians) / (mads + eps)

    return X_out
