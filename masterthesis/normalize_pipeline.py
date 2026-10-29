#!/usr/bin/env python
"""
normalize_pipeline.py — Single entry point for all batch correction methods.

Applies optional preprocessing steps (--pre) followed by a batch correction
method (--method). Everything shared across methods (data loading, PCA, saving)
lives here. Method-specific logic lives in the companion modules:

  tvn.py           TVN class + apply()
  harmony.py       apply()
  combat.py        run_combat(), apply_combat(), apply()
  scvi_method.py   apply()     (named to avoid shadowing the scvi package)
  mnn.py           apply()     (Scanorama)
  fastmnn.py       apply()     (fastMNN — pure Python MNN correction)
  mad.py           normalize() (preprocessing)
  sphering.py      apply()     (preprocessing)

Output directory
----------------
Named by joining pre-steps and method with underscores:

  --method harmony                     →  <input-dir>/harmony/
  --pre mad --method harmony           →  <input-dir>/mad_harmony/
  --pre mad sphering --method tvn      →  <input-dir>/mad_sphering_tvn/
  --method none                        →  <input-dir>/none/

Output files (same for all methods):
  Z_tr_corr.npy    Processed train+val embeddings   [N_tr+N_va, n_dim]
  meta_tr.pq       Aligned metadata for train+val
  Z_te_corr.npy    Processed test embeddings         [N_te, n_dim]
  meta_te.pq       Aligned metadata for test
  pca.pkl          Fitted PCA object (joblib) — for inverse_transform

Usage
-----
  conda activate mm_cpu

  python masterthesis/normalize_pipeline.py \\
      --input-dir checkpoints/run/channelwise_embeddings \\
      --meta-path source_3_w_neg.pq \\
      --method harmony

  python masterthesis/normalize_pipeline.py \\
      --input-dir checkpoints/run/channelwise_embeddings \\
      --meta-path source_3_w_neg.pq \\
      --pre mad --method harmony

  python masterthesis/normalize_pipeline.py \\
      --input-dir checkpoints/run/channelwise_embeddings \\
      --meta-path source_3_w_neg.pq \\
      --pre mad sphering --method tvn
"""

import os, sys
# -- HPC: cap BLAS threads to prevent OpenBLAS >128-thread crash --
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"
# -- Block the TensorFlow import to prevent a protobuf crash --
# (scanpy → louvain → TF → duplicate xla_data.proto registration = abort);
# none of the corrections needs TensorFlow.
import types, importlib.machinery
_tf_dummy = types.ModuleType("tensorflow")
_tf_dummy.__version__ = "0.0.0"
_tf_dummy.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
sys.modules["tensorflow"] = _tf_dummy

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import detect_batch_keys, load_split, scale_and_pca


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Composable preprocessing + batch correction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- Common ---
    p.add_argument("--input-dir", required=True,
                   help="Directory containing train/val/test channelwise_embeddings.npz")
    p.add_argument("--meta-path", default="source_3_w_neg.pq")
    p.add_argument("--control-key", default="DMSO")
    p.add_argument("--batch-keys", nargs="+", default=None,
                   choices=["batch", "plate", "well"])
    p.add_argument("--n-components", type=int, default=50,
                   help="PCA components in the final embedding")

    # --- Preprocessing ---
    p.add_argument("--pre", nargs="*", default=[],
                   choices=["mad", "sphering"],
                   help="Preprocessing steps applied in order before batch correction")
    p.add_argument("--eps-mad", type=float, default=1e-8,
                   help="[mad] Epsilon for MAD denominator")
    p.add_argument("--sphering-method", default="ZCA-cor",
                   choices=["ZCA", "PCA", "ZCA-cor", "PCA-cor"],
                   help="[sphering] Whitening method")
    p.add_argument("--eps-sphering", type=float, default=1e-6,
                   help="[sphering] Epsilon for numerical stability")

    # --- Batch correction method ---
    p.add_argument("--method", default="none",
                   choices=["none", "tvn", "harmony", "combat", "scvi", "mnn",
                            "fastmnn", "seurat"],
                   help="Batch correction method")

    # --- TVN ---
    p.add_argument("--no-pca-feat", action="store_true",
                   help="[tvn] Disable internal PCA preprocessing in TVN")
    p.add_argument("--dmso-subsample", type=int, default=None,
                   help="[tvn, E1] Cap DMSO wells per batch used to fit the "
                        "target covariance. Default: use all.")
    p.add_argument("--dmso-subsample-seed", type=int, default=42,
                   help="[tvn, E1] Seed for --dmso-subsample.")
    p.add_argument("--save-fitted", default=None,
                   help="[tvn, E4] Path to pickle the fitted TVN state for "
                        "later cross-source reuse.")
    p.add_argument("--load-fitted", default=None,
                   help="[tvn, E4] Load a previously saved TVN state and "
                        "skip the fit step. Used for cross-source portability.")

    # --- ComBat ---
    p.add_argument("--no-eb", action="store_true",
                   help="[combat] Disable Empirical Bayes")
    p.add_argument("--mean-only", action="store_true",
                   help="[combat] Adjust means only, not variance")
    p.add_argument("--ref-batch", default=None,
                   help="[combat] Reference batch label")

    # --- scVI ---
    p.add_argument("--n-latent", type=int, default=50,
                   help="[scvi] Latent dimensionality")
    p.add_argument("--n-hidden", type=int, default=256,
                   help="[scvi] Encoder/decoder hidden size")
    p.add_argument("--n-layers", type=int, default=2,
                   help="[scvi] Encoder/decoder depth")
    p.add_argument("--max-epochs", type=int, default=200,
                   help="[scvi] Training epochs")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="[scvi] Learning rate")
    p.add_argument("--seed", type=int, default=42)

    # --- MNN / Scanorama ---
    p.add_argument("--k", type=int, default=20,
                   help="[mnn] Nearest neighbours for MNN search")
    p.add_argument("--sigma", type=float, default=15.0,
                   help="[mnn] Gaussian kernel bandwidth")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="[mnn] Minimum fraction of MNN pairs required")
    p.add_argument("--no-approx", action="store_true",
                   help="[mnn] Disable approximate nearest-neighbour search")
    p.add_argument("--no-geosketch", action="store_true",
                   help="[mnn] Disable geosketch subsampling")
    p.add_argument("--geosketch-max", type=int, default=5000,
                   help="[mnn] Max cells per batch for geosketch")

    # --- fastMNN ---
    p.add_argument("--fastmnn-k", type=int, default=20,
                   help="[fastmnn] Nearest neighbours for MNN search")
    p.add_argument("--fastmnn-sigma", type=float, default=1.0,
                   help="[fastmnn] Gaussian kernel bandwidth for correction")

    # --- Seurat (R) ---
    p.add_argument("--seurat-dims", type=int, default=30,
                   help="[seurat] #dims for the RPCA reduction (scib default 30)")

    # --- Output ---
    p.add_argument("--out-subdir", default=None,
                   help="Override auto-generated output subdir name")

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    args = parse_args()
    root = Path(args.input_dir)

    # --- Batch keys ---
    if args.batch_keys is None:
        args.batch_keys = detect_batch_keys(root)
        if args.batch_keys is None:
            raise SystemExit(
                "Could not auto-detect batch keys. "
                "Pass --batch-keys explicitly (e.g. --batch-keys batch)."
            )
    batch_key = args.batch_keys[0]

    # --- Output directory ---
    if args.out_subdir:
        subdir_name = args.out_subdir
    else:
        parts = list(args.pre or []) + [args.method]
        subdir_name = "_".join(parts)
    out_dir = root / subdir_name

    print(f"\nPipeline: pre={args.pre!r}  method={args.method!r}")
    print(f"Output dir: {out_dir}\n")

    # --- Load raw embeddings ---
    source_3 = pd.read_parquet(args.meta_path)
    source_3["Metadata_Sample_ID"] = source_3["Metadata_Sample_ID"].astype(str)

    X_tr, meta_tr = load_split(root / "train_channelwise_embeddings.npz",
                               source_3, args.control_key)
    X_va, meta_va = load_split(root / "validation_channelwise_embeddings.npz",
                               source_3, args.control_key)
    X_te, meta_te = load_split(root / "test_channelwise_embeddings.npz",
                               source_3, args.control_key)
    print(f"Loaded: train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")

    # --- Preprocessing ---
    if args.pre:
        print(f"\n--- Preprocessing: {args.pre} ---")
        for step in args.pre:
            if step == "mad":
                from mad import normalize as mad_normalize
                print(f"  MAD normalizing per '{batch_key}' ...")
                X_tr = mad_normalize(X_tr, meta_tr, batch_key, args.eps_mad).astype(np.float32)
                X_va = mad_normalize(X_va, meta_va, batch_key, args.eps_mad).astype(np.float32)
                X_te = mad_normalize(X_te, meta_te, batch_key, args.eps_mad).astype(np.float32)
            elif step == "sphering":
                from sphering import apply as sphering_apply
                X_tr, X_va, X_te = sphering_apply(
                    X_tr, X_va, X_te, meta_tr,
                    method=args.sphering_method, eps=args.eps_sphering,
                )

    # --- Batch correction ---
    print(f"\n--- Batch correction: {args.method} ---")

    if args.method == "none":
        Z_tr, Z_va, Z_te, n_comp, ev, pca = scale_and_pca(
            X_tr, X_va, X_te, args.n_components)
        print(f"  PCA: {X_tr.shape[1]}D -> {n_comp}D  (explained variance: {ev:.3f})")
        Z_trainval    = np.vstack([Z_tr, Z_va])
        meta_trainval = pd.concat([meta_tr, meta_va], ignore_index=True)
        Z_te_corr     = Z_te
        meta_te_out   = meta_te

    elif args.method == "tvn":
        from tvn import apply as tvn_apply
        Z_trainval, meta_trainval, Z_te_corr, meta_te_out, pca = tvn_apply(
            X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
            batch_key, args.n_components,
            no_pca_feat=args.no_pca_feat,
            dmso_subsample=args.dmso_subsample,
            dmso_subsample_seed=args.dmso_subsample_seed,
            save_fitted=args.save_fitted,
            load_fitted=args.load_fitted,
        )

    elif args.method == "harmony":
        from harmony import apply as harmony_apply
        Z_trainval, meta_trainval, Z_te_corr, meta_te_out, pca = harmony_apply(
            X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
            args.batch_keys, args.n_components,
        )

    elif args.method == "combat":
        from combat import apply as combat_apply
        out_dir.mkdir(parents=True, exist_ok=True)  # needed before combat saves estimates.pkl
        Z_trainval, meta_trainval, Z_te_corr, meta_te_out, pca = combat_apply(
            X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
            args.batch_keys, args.n_components,
            eb=not args.no_eb, mean_only=args.mean_only, ref_batch=args.ref_batch,
            out_dir=out_dir,
        )

    elif args.method == "scvi":
        from scvi_method import apply as scvi_apply
        Z_trainval, meta_trainval, Z_te_corr, meta_te_out, pca = scvi_apply(
            X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
            batch_key, args.n_components,
            n_latent=args.n_latent, n_hidden=args.n_hidden,
            n_layers=args.n_layers, max_epochs=args.max_epochs,
            lr=args.lr, seed=args.seed,
        )

    elif args.method == "mnn":
        from mnn import apply as mnn_apply
        Z_trainval, meta_trainval, Z_te_corr, meta_te_out, pca = mnn_apply(
            X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
            batch_key, args.n_components,
            k=args.k, sigma=args.sigma, alpha=args.alpha,
            approx=not args.no_approx,
            geosketch=not args.no_geosketch,
            geosketch_max=args.geosketch_max,
        )

    elif args.method == "fastmnn":
        from fastmnn import apply as fastmnn_apply
        Z_trainval, meta_trainval, Z_te_corr, meta_te_out, pca = fastmnn_apply(
            X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
            batch_key, args.n_components,
            k=args.fastmnn_k, sigma=args.fastmnn_sigma,
        )

    elif args.method == "seurat":
        from seurat_method import apply as seurat_apply
        Z_trainval, meta_trainval, Z_te_corr, meta_te_out, pca = seurat_apply(
            X_tr, X_va, X_te, meta_tr, meta_va, meta_te,
            batch_key, args.n_components,
            dims=args.seurat_dims,
        )

    # --- Save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, out_dir / "pca.pkl")
    np.save(out_dir / "Z_tr_corr.npy", Z_trainval)
    meta_trainval.reset_index(drop=True).to_parquet(out_dir / "meta_tr.pq")
    np.save(out_dir / "Z_te_corr.npy", Z_te_corr)
    meta_te_out.reset_index(drop=True).to_parquet(out_dir / "meta_te.pq")

    print(f"\nSaved Z_tr_corr {Z_trainval.shape}, Z_te_corr {Z_te_corr.shape} -> {out_dir}")
    print("\nNext step:")
    print(f"  python masterthesis/eval_all.py \\")
    print(f"      --embeddings-dir {root} \\")
    print(f"      --meta-path {args.meta_path} \\")
    print(f"      --out-json {root}/eval_all.json")
