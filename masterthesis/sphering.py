"""
sphering.py — Global ZCA/PCA whitening (Spherize).

Called by normalize_pipeline.py. Not a standalone script.

Uses the Spherize implementation from Arevalo et al. (2024) Nature Communications.
Requires the 2023_Arevalo_NatComm_BatchCorrection repo cloned next to masterthesis/.
"""

import contextlib
import sys
import numpy as np
import numpy.linalg as _la
from pathlib import Path

_AREVALO_ROOT = Path(__file__).parent.parent / "2023_Arevalo_NatComm_BatchCorrection"


@contextlib.contextmanager
def _econ_svd():
    """Force economy SVD for the duration of Spherize.fit.

    Arevalo's Spherize calls np.linalg.svd(full_matrices=True). On a DMSO matrix of
    n rows that allocates an n x n U (~200k x 200k float64 = O(100 GiB)), which
    OOMs on any realistic node. For n >= d the economy SVD returns the IDENTICAL
    Vt/Sigma, hence an identical whitening, without materialising U.

    tvn_lobo.py applies the same guard globally; here it is scoped to the fit so
    the patch cannot leak into unrelated callers.
    """
    orig = _la.svd

    def _patched(a, full_matrices=True, **kw):
        return orig(a, full_matrices=False, **kw)

    _la.svd = _patched
    np.linalg.svd = _patched
    try:
        yield
    finally:
        _la.svd = orig
        np.linalg.svd = orig


def apply(X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray,
          meta_tr,
          method: str = "ZCA-cor",
          eps: float = 1e-6,
          out_dir=None):
    """
    Global sphering fit on train DMSO controls, applied to all splits.

    Parameters
    ----------
    method   : ZCA | PCA | ZCA-cor | PCA-cor  (ZCA-cor recommended for DL embeddings)
    eps      : fudge factor for numerical stability
    out_dir  : if given, saves spherer.npz (weights W) for reproducibility

    Returns (X_tr_sph, X_va_sph, X_te_sph) as float32.
    """
    sys.path.insert(0, str(_AREVALO_ROOT))
    try:
        from pycytominer.operations import Spherize
    except ImportError as e:
        raise SystemExit(
            f"Could not import Spherize from {_AREVALO_ROOT}.\n"
            "Clone 2023_Arevalo_NatComm_BatchCorrection next to the masterthesis dir.\n"
            f"  Error: {e}"
        )

    dmso_mask = meta_tr["is_control"].values
    print(f"  [sphering] Spherize(method={method!r}, eps={eps}) "
          f"on {dmso_mask.sum()} train DMSO wells ...")

    spherer = Spherize(epsilon=eps, method=method)
    with _econ_svd():                       # else full_matrices=True OOMs — see above
        spherer.fit(X_tr[dmso_mask])

    X_tr_sph = spherer.transform(X_tr).astype(np.float32)
    X_va_sph = spherer.transform(X_va).astype(np.float32)
    X_te_sph = spherer.transform(X_te).astype(np.float32)

    if out_dir is not None:
        np.savez_compressed(Path(out_dir) / "spherer.npz", W=spherer.W)

    return X_tr_sph, X_va_sph, X_te_sph
