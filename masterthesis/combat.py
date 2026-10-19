"""
combat.py — ComBat batch correction.

Called by normalize_pipeline.py. Not a standalone script.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from neuroCombat import neuroCombat, neuroCombatFromTraining

from common import scale_and_pca


def run_combat(Z: np.ndarray, batch: pd.Series,
               eb: bool, mean_only: bool, ref_batch):
    """
    Fit ComBat on Z [N, D]. neuroCombat expects (D, N).
    Returns (Z_corrected [N, D], estimates dict).
    """
    covars = pd.DataFrame({"batch": batch.astype(str).values})
    result = neuroCombat(
        dat=Z.T,
        covars=covars,
        batch_col="batch",
        eb=eb,
        parametric=True,
        mean_only=mean_only,
        ref_batch=ref_batch,
    )
    return result["data"].T, result["estimates"]


def apply_combat(Z: np.ndarray, batch: pd.Series, estimates: dict):
    """Apply pre-fitted ComBat parameters to a new split."""
    result = neuroCombatFromTraining(
        dat=Z.T,
        batch=batch.astype(str).values,
        estimates=estimates,
    )
    return result["data"].T


def apply(X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
          batch_keys, n_components,
          eb=True, mean_only=False, ref_batch=None,
          out_dir=None):
    """
    StandardScaler + PCA (fit on train only), then ComBat on train+val combined.

    Test is corrected via neuroCombatFromTraining ONLY for batches seen at fit
    time. For batches disjoint from train+val (e.g. split=batches) ComBat has no
    per-batch parameters, so the returned test array is the UNCORRECTED raw PCA —
    ComBat is transductive and CANNOT be applied to an unseen batch. Do not treat
    the test output as a held-out / deployment result for ComBat.

    If out_dir is given, saves estimates.pkl for reproducibility.

    Returns (Z_trainval, meta_trainval, Z_te, meta_te, pca).
    """
    batch_key = batch_keys[0]
    Z_tr, Z_va, Z_te, n_comp, ev, pca = scale_and_pca(
        X_tr, X_va, X_te, n_components)
    print(f"  [combat] PCA: {X_tr.shape[1]}D -> {n_comp}D  (explained variance: {ev:.3f})")

    Z_trainval    = np.vstack([Z_tr, Z_va])
    meta_trainval = pd.concat([meta_tr, meta_va], ignore_index=True)

    print(f"  [combat] Fitting on train+val ({Z_trainval.shape[0]} samples) "
          f"batch_key='{batch_key}' | eb={eb} | mean_only={mean_only} ...")
    Z_trainval_corr, estimates = run_combat(
        Z_trainval, meta_trainval[batch_key], eb, mean_only, ref_batch)

    print(f"  [combat] Applying to test ({Z_te.shape[0]} samples) ...")
    try:
        Z_te_corr = apply_combat(Z_te, meta_te[batch_key], estimates)
    except ValueError as exc:
        # Expected whenever the test batches are disjoint from train+val (always,
        # under split=batches): ComBat estimates per-batch parameters and has none
        # for a batch it never saw. We return the raw PCA so the (valid) train+val
        # correction still completes — but the test array is NOT ComBat-corrected.
        # It must NOT be treated as a held-out/deployment result (ComBat is
        # transductive). See the apply() docstring.
        print(f"  [combat] NOTE: ComBat cannot transform test batches it was not "
              f"fit on ({exc}).")
        print("  [combat] >>> TEST OUTPUT IS UNCORRECTED RAW PCA (not ComBat) <<<")
        Z_te_corr = Z_te

    if out_dir is not None:
        with open(Path(out_dir) / "estimates.pkl", "wb") as f:
            pickle.dump(estimates, f)

    return Z_trainval_corr, meta_trainval, Z_te_corr, meta_te, pca
