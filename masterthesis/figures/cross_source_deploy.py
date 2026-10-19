#!/usr/bin/env python
"""Cross-source deployment: does a Source-3 correction recipe transfer to Source-1?

This is the "unseen SOURCE" analogue of tvn_lobo.py's "unseen BATCH" test. Where LOBO
holds out one batch of the SAME source, this freezes the whole Source-3 recipe and drops
Source-1 batches on it — the real production question ("a plate from a new lab arrives").

PREREQUISITE (why this needs a GPU extraction first): a fitted TVN/sphering transform
lives in its ENCODER's feature space. The S3 and S1 canonical runs are separately-trained
weight sets, so their 2048-D / 1920-D spaces are not aligned — you cannot apply an
S3-fitted whitening to S1-encoder embeddings. To put both sources in ONE space, S1 images
must be run through the S3 weights (gpu_cross_source_extract.slrm). This script consumes
those S1-through-S3 embeddings as the "deploy" set.

Design (n = number of S1 batches, = 6 for Source 1):
    R  = ALL Source-3   (the "batches you already have" — frozen recipe source)
    H  = each Source-1 batch b, encoded through the SAME S3 weights
  * fit the corrector's recipe (scaler + feature-PCA + whitening + final 50-D PCA) on R,
  * transform H using H's OWN DMSO controls (the deployable contract — never fit on H),
  * score H only, vs a RAW baseline that is also R-fit-PCA projected onto H.
This yields one paired (method - raw) delta per S1 batch; we report mean + bootstrap CI
OVER S1 BATCHES, exactly as tvn_lobo.py does over the held-out batches — so the two panels
(within-S1 LOBO vs cross-source deploy) are read on the same axes.

The correctors and scoring are imported verbatim from tvn_lobo.py so the numbers are
directly comparable; only the fold definition differs (R is fixed = all S3, H iterates
over S1 batches, and R/H come from different metadata files but the same embedding space).

Runtime: R = all of Source 3 is large and the corrector is refit per S1 batch (6x). Prefer
the cluster; use --ref-clean-only to drop S3's augmented train split, and --dmso-subsample
to cap the TVN fit.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "masterthesis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import tvn_lobo`

# Reuse the LOBO correctors + scoring VERBATIM (incl. the econ-SVD monkeypatch that
# tvn_lobo installs at import) so cross-source deltas match the within-source LOBO deltas.
import tvn_lobo as L  # noqa: E402
from masterthesis.common import load_split  # noqa: E402

BATCH_KEY = L.BATCH_KEY  # "batch"

# Canonical S3 references (the frozen-recipe source) and their metadata.
REF = {
    "resnet": REPO / "checkpoints/stoic-snowball-398_9llyzmze/channelwise_embeddings",
    "mae":    REPO / "checkpoints/honest-snowflake-408_x79nwnqn/channelwise_embeddings",
}
REF_META = REPO / "source_3_w_neg.pq"
DEPLOY_META = REPO / "source_1_w_neg.pq"
OUT = REPO / "thesis_figures" / "tvn_holdout" / "cross_source"


def _load_pool(ce_dir: Path, meta_path: Path, splits) -> tuple[np.ndarray, pd.DataFrame]:
    """Pool the requested splits of one channelwise-embeddings dir into (X, meta)."""
    src = pd.read_parquet(meta_path)
    parts = []
    for split in splits:
        npz = ce_dir / f"{split}_channelwise_embeddings.npz"
        if not npz.exists():
            print(f"[warn] missing {npz.name} in {ce_dir} — skipping", flush=True)
            continue
        X, m = load_split(npz, src, "DMSO")
        parts.append((X, m))
    if not parts:
        raise FileNotFoundError(f"no embedding splits found under {ce_dir}")
    X = np.vstack([p[0] for p in parts])
    meta = pd.concat([p[1] for p in parts], ignore_index=True)
    return X, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["resnet", "mae"], required=True,
                    help="which S3 reference/recipe to deploy")
    ap.add_argument("--deploy-ce", type=Path, required=True,
                    help="channelwise_embeddings dir of S1 images run through the S3 weights "
                         "(produced by gpu_cross_source_extract.slrm)")
    ap.add_argument("--ref-ce", type=Path, default=None,
                    help="S3 reference channelwise_embeddings dir (default: canonical for --encoder)")
    ap.add_argument("--ref-meta", type=Path, default=REF_META)
    ap.add_argument("--deploy-meta", type=Path, default=DEPLOY_META)
    ap.add_argument("--methods", nargs="*", default=["raw", "tvn", "sphering", "mad"])
    ap.add_argument("--ref-clean-only", action="store_true",
                    help="use only S3 validation+test as the reference (drops the augmented "
                         "train split) so the reference matches the CLEAN S1 deploy extraction")
    ap.add_argument("--dmso-subsample", type=int, default=None,
                    help="cap DMSO wells/batch in the TVN fit on the reference (speed)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    ref_ce = args.ref_ce or REF[args.encoder]
    run_name = ref_ce.resolve().parents[0].name
    out_dir = args.out_dir or (OUT / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_splits = ("validation", "test") if args.ref_clean_only else ("train", "validation", "test")
    X_R, meta_R = _load_pool(ref_ce, args.ref_meta, ref_splits)
    X_H_all, meta_H_all = _load_pool(args.deploy_ce, args.deploy_meta, ("train", "validation", "test"))
    if X_R.shape[1] != X_H_all.shape[1]:
        raise ValueError(
            f"feature-dim mismatch: reference {X_R.shape[1]}-D vs deploy {X_H_all.shape[1]}-D. "
            "The deploy embeddings must come from the SAME encoder weights as the reference "
            "(run S1 images through the S3 checkpoint) — see gpu_cross_source_extract.slrm.")

    s1_batches = sorted(meta_H_all[BATCH_KEY].astype(int).unique())
    print(f"[pool] R(S3 {'clean' if args.ref_clean_only else 'all'})={X_R.shape}  "
          f"H(S1-through-S3)={X_H_all.shape}  S1 batches={s1_batches}", flush=True)

    rows = []
    for b in s1_batches:
        t0 = time.time()
        H = meta_H_all[BATCH_KEY].astype(int).to_numpy() == b
        X_H, meta_H = X_H_all[H], meta_H_all.loc[H].reset_index(drop=True)
        if meta_H["is_control"].sum() == 0:
            print(f"[skip b={b}] S1 batch has no DMSO controls — deployable transform needs them")
            continue
        for name in args.methods:
            Z_R, Z_H = L.CORRECTORS[name](X_R, meta_R, X_H, meta_H, BATCH_KEY,
                                          dmso_subsample=args.dmso_subsample)
            s = L.score_holdout(Z_R, meta_R, Z_H, meta_H)
            rows.append(dict(batch=int(b), method=name, **s))
            del Z_R, Z_H
        r = {x["method"]: x for x in rows if x["batch"] == b}
        base = args.methods[0] if args.methods[0] != "raw" else "raw"
        print(f"[s1 b={b:2d}] n={r['raw']['n_wells']:4d}  "
              f"raw(AU {r['raw']['auroc']:.3f} iL {r['raw']['ilisi']:.2f})  "
              + "  ".join(f"{m}(AU {r[m]['auroc']:.3f} iL {r[m]['ilisi']:.2f})"
                          for m in args.methods if m != "raw" and m in r)
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        del X_H

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cross_source_per_batch.csv", index=False)

    # paired deltas (method - raw) per S1 batch, mean + bootstrap CI over the S1 batches
    piv = df.pivot(index="batch", columns="method")
    md = (f"# Cross-source deployment ({run_name} recipe -> Source-1 batches)\n\n"
          f"_S3 recipe frozen, applied to n={piv.shape[0]} Source-1 batches (each transformed via "
          f"its OWN DMSO). Reference = Source-3 {'val+test (clean)' if args.ref_clean_only else 'all splits'}. "
          f"Well-aggregated. dmso_subsample={args.dmso_subsample}. Mean Δ (method − raw) "
          f"[95% bootstrap CI over S1 batches]. cross_source_deploy.py_\n\n")
    read = {"auroc": "bio preserved (want ≥0)", "bacc": "bio: compounds classifiable (want ≥0)",
            "ilisi": "S1 batch mixes into S3 (want +)",
            "sil": "mixing (want −)", "kbet": "mixing (want −)"}
    for method in [m for m in args.methods if m != "raw"]:
        md += f"\n### {method} − raw  (mean Δ [95% bootstrap CI over S1 batches])\n\n"
        md += "| metric | mean Δ | 95% CI | reading |\n|---|---|---|---|\n"
        for k in ("auroc", "bacc", "ilisi", "sil", "kbet"):
            dvals = (piv[k][method] - piv[k]["raw"]).to_numpy()
            mean, lo, hi = L.boot_ci(dvals)
            md += f"| Δ {k} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {read[k]} |\n"
    md += ("\n_Compare against the within-Source-1 LOBO (tvn_lobo.py on the S1-native encoder): "
           "if the frozen S3 recipe reaches similar Δ iLISI at similar Δ AUROC, the correction "
           "deploys across sources; if it collapses, the recipe is source-specific._\n")
    (out_dir / "cross_source_summary.md").write_text(md)
    print("\n" + md)
    print(f"[wrote] {out_dir/'cross_source_per_batch.csv'}\n[wrote] {out_dir/'cross_source_summary.md'}")


if __name__ == "__main__":
    main()
