"""seurat_method.py — Seurat (R) reciprocal-PCA batch integration.

Python bridge to seurat_integrate.R. Mirrors harmony.py: StandardScaler + PCA
(fit on train only), then Seurat v5 RPCA integration of the train+val 50-PC
matrix, correcting by the batch key. Test receives the PCA projection only
(Seurat integration is transductive; test is unused in the benchmark).

The R step runs Seurat v5's RPCAIntegration and returns the corrected reduction
(N x dims, default 30 = scib default). The bridge is a CSV round-trip so no
exotic R packages (arrow/RcppCNPy) are needed.

Requires an environment with BOTH the Python deps (numpy/pandas/sklearn) AND
Rscript + r-seurat on PATH — e.g. `conda activate seurat`. Called by
normalize_pipeline.py; not a standalone script.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import scale_and_pca

_R_SCRIPT = Path(__file__).parent / "seurat_integrate.R"


def apply(X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
          batch_key, n_components, rscript=None, dims=30, work_dir=None,
          keep_work=False):
    """Returns (Z_trainval_corr, meta_trainval, Z_te, meta_te, pca)."""
    rscript = rscript or os.environ.get("SEURAT_RSCRIPT", "Rscript")

    Z_tr, Z_va, Z_te, n_comp, ev, pca = scale_and_pca(
        X_tr, X_va, X_te, n_components)
    print(f"  [seurat] PCA: {X_tr.shape[1]}D -> {n_comp}D  (explained variance: {ev:.3f})")

    Z_trainval_pca = np.vstack([Z_tr, Z_va]).astype(np.float64)
    meta_trainval  = pd.concat([meta_tr, meta_va], ignore_index=True)
    batch = meta_trainval[batch_key].astype(str).to_numpy()
    n_batches = len(np.unique(batch))
    print(f"  [seurat] RPCA integration on {Z_trainval_pca.shape[0]} cells, "
          f"{n_batches} '{batch_key}'-groups, dims=1:{dims} ...", flush=True)

    tmp = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="seurat_bridge_",
                         dir=os.environ.get("SEURAT_WORKDIR") or None))
    tmp.mkdir(parents=True, exist_ok=True)
    emb_in  = tmp / "emb_in.csv"
    bat_in  = tmp / "batch_in.csv"
    emb_out = tmp / "emb_corr.csv"

    pd.DataFrame(Z_trainval_pca).to_csv(emb_in, header=False, index=False)
    pd.DataFrame({"batch": batch}).to_csv(bat_in, index=False)

    cmd = [rscript, str(_R_SCRIPT), str(emb_in), str(bat_in), str(emb_out), str(dims)]
    print(f"  [seurat] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    Z_corr = pd.read_csv(emb_out, header=None).to_numpy()
    if Z_corr.shape[0] != Z_trainval_pca.shape[0]:
        raise RuntimeError(
            f"Seurat returned {Z_corr.shape[0]} rows, expected {Z_trainval_pca.shape[0]}")
    print(f"  [seurat] corrected embedding: {Z_corr.shape}")

    if not keep_work:
        shutil.rmtree(tmp, ignore_errors=True)

    return Z_corr, meta_trainval, Z_te, meta_te, pca
