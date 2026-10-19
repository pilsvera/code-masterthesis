"""
tvn.py — Typical Variation Normalization (TVN).

Called by normalize_pipeline.py. Not a standalone script.

Reference: Arevalo et al. 2024 Nature Communications,
"Evaluating batch correction methods for image-based cell profiling".
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TVN:
    """
    Typical Variation Normalization.

    Fits a global StandardScaler + full PCA on pooled train DMSO controls,
    then maps each batch's covariance to a shared target covariance via
    Mahalanobis whitening.
    """

    def __init__(self, pcafeat: bool = True):
        self.pcafeat = pcafeat
        if self.pcafeat:
            self.pca = PCA()       # full PCA — no dimension reduction here
        self.scaler = StandardScaler()

    def fit(self, X_neg: np.ndarray, batch_neg: np.ndarray,
            dmso_subsample: int | None = None, seed: int = 42):
        """Fit on pooled DMSO controls across all train batches.

        If `dmso_subsample` is given, subsample stratified per batch — each
        batch contributes at most `dmso_subsample` wells. Batches with fewer
        wells contribute all of them.
        """
        if dmso_subsample is not None and dmso_subsample > 0:
            rng = np.random.default_rng(seed)
            keep = []
            for b in np.unique(batch_neg):
                idx_b = np.where(batch_neg == b)[0]
                n_keep = min(dmso_subsample, len(idx_b))
                keep.append(rng.choice(idx_b, size=n_keep, replace=False))
            keep = np.concatenate(keep)
            X_neg     = X_neg[keep]
            batch_neg = batch_neg[keep]

        X_neg = self.scaler.fit_transform(X_neg)
        X_neg = self.pca.fit_transform(X_neg) if self.pcafeat else X_neg

        X_neg_transformed = np.empty_like(X_neg, dtype=float)
        for b in np.unique(batch_neg):
            idx = np.where(batch_neg == b)[0]
            pca_scaler = StandardScaler()
            X_neg_transformed[idx] = pca_scaler.fit_transform(X_neg[idx])

        self.target_cov = self._cov(X_neg_transformed)

    def transform(self, X: np.ndarray, batch: np.ndarray,
                  X_neg: np.ndarray, batch_neg: np.ndarray) -> np.ndarray:
        """
        Apply TVN to X.

        For each batch, computes source covariance from that batch's DMSO
        controls and applies the covariance-matching transform.
        """
        common_batches = np.intersect1d(np.unique(batch), np.unique(batch_neg))
        missing = np.setdiff1d(np.unique(batch), np.unique(batch_neg))
        if missing.size:
            raise ValueError(
                f"TVN.transform: {missing.size} batch(es) have no DMSO controls and "
                f"cannot be corrected: {missing.tolist()}. Every batch must include "
                f"DMSO (negative-control) wells; without them these rows would be "
                f"returned as uninitialised memory. Provide controls or drop the batch."
            )
        X_out = np.empty(X.shape, dtype=float)

        for b in common_batches:
            idx_neg = np.where(batch_neg == b)[0]
            X_neg_b = self.scaler.transform(X_neg[idx_neg])
            if self.pcafeat:
                X_neg_b = self.pca.transform(X_neg_b)
            pca_scaler = StandardScaler()
            X_neg_b = pca_scaler.fit_transform(X_neg_b)
            src_cov = self._cov(X_neg_b)

            idx = np.where(batch == b)[0]
            X_b = self.scaler.transform(X[idx])
            if self.pcafeat:
                X_b = self.pca.transform(X_b)
            X_b = pca_scaler.transform(X_b)
            X_b = X_b @ linalg.fractional_matrix_power(src_cov, -0.5)
            X_b = X_b @ linalg.fractional_matrix_power(self.target_cov, 0.5)
            X_out[idx] = X_b

        return X_out.astype(np.float32)

    def save(self, path: Path) -> None:
        """Pickle the fitted scaler / PCA / target_cov to `path`."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "pcafeat":    self.pcafeat,
            "scaler":     self.scaler,
            "pca":        self.pca if self.pcafeat else None,
            "target_cov": self.target_cov,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "TVN":
        """Load fitted state previously written by `save()`."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(pcafeat=state["pcafeat"])
        obj.scaler     = state["scaler"]
        if state["pcafeat"]:
            obj.pca = state["pca"]
        obj.target_cov = state["target_cov"]
        return obj

    @staticmethod
    def _cov(X: np.ndarray) -> np.ndarray:
        return np.cov(X, rowvar=False, ddof=1) + 0.5 * np.eye(X.shape[1])


def apply(X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
          batch_key, n_components, no_pca_feat=False,
          dmso_subsample: int | None = None, dmso_subsample_seed: int = 42,
          save_fitted: str | None = None, load_fitted: str | None = None):
    """
    TVN batch correction followed by PCA dimensionality reduction.

    TVN is fit on train DMSO controls, transforms all splits.
    PCA is fit on train only (no val/test leakage).

    `dmso_subsample` caps the per-batch DMSO count used to fit the target
    covariance.

    Cross-source portability: pass `save_fitted=<path.pkl>` after a fit
    on source A, then re-run on source B with `load_fitted=<same path>` and
    the source-B DMSO controls; the target covariance is held fixed from A
    while per-batch source covariance is taken from B's controls.

    Returns (Z_trainval, meta_trainval, Z_te, meta_te, pca).
    """
    ctrl_tr = meta_tr["is_control"].values
    ctrl_va = meta_va["is_control"].values
    ctrl_te = meta_te["is_control"].values

    X_neg_tr = X_tr[ctrl_tr];  b_neg_tr = meta_tr.loc[ctrl_tr, batch_key].to_numpy()
    X_neg_va = X_va[ctrl_va];  b_neg_va = meta_va.loc[ctrl_va, batch_key].to_numpy()
    X_neg_te = X_te[ctrl_te];  b_neg_te = meta_te.loc[ctrl_te, batch_key].to_numpy()

    if load_fitted is not None:
        print(f"  [tvn] Loading fitted state from {load_fitted} ...")
        tvn = TVN.load(load_fitted)
    else:
        if dmso_subsample is not None:
            print(f"  [tvn] Fitting on ≤{dmso_subsample} DMSO wells per batch "
                  f"(of {ctrl_tr.sum()} total, seed={dmso_subsample_seed}) ...")
        else:
            print(f"  [tvn] Fitting on {ctrl_tr.sum()} train DMSO wells ...")
        tvn = TVN(pcafeat=not no_pca_feat)
        tvn.fit(X_neg_tr, b_neg_tr,
                dmso_subsample=dmso_subsample, seed=dmso_subsample_seed)

    if save_fitted is not None:
        print(f"  [tvn] Saving fitted state to {save_fitted} ...")
        tvn.save(save_fitted)

    print("  [tvn] Transforming train, val, test ...")
    X_tr_tvn = tvn.transform(X_tr, meta_tr[batch_key].to_numpy(), X_neg_tr, b_neg_tr)
    X_va_tvn = tvn.transform(X_va, meta_va[batch_key].to_numpy(), X_neg_va, b_neg_va)
    X_te_tvn = tvn.transform(X_te, meta_te[batch_key].to_numpy(), X_neg_te, b_neg_te)

    X_trainval    = np.vstack([X_tr_tvn, X_va_tvn])
    meta_trainval = pd.concat([meta_tr, meta_va], ignore_index=True)

    n_comp = min(n_components, X_tr_tvn.shape[1], X_tr_tvn.shape[0])
    print(f"  [tvn] PCA({n_comp}D) fit on train ({X_tr_tvn.shape[0]} samples) ...")
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_tr_tvn)
    Z_trainval = pca.transform(X_trainval)
    Z_te       = pca.transform(X_te_tvn)
    ev = float(pca.explained_variance_ratio_.sum())
    print(f"  [tvn] PCA: {X_tr_tvn.shape[1]}D -> {n_comp}D  (explained variance: {ev:.3f})")

    return Z_trainval, meta_trainval, Z_te, meta_te, pca
