#!/usr/bin/env python
"""
make_all13_split.py — build an "all batches in the fit" input dir.

normalize_pipeline fits every correction on TRAIN only (val/test are projected,
test held out). For the in-sample integration-quality probe we want the
correction to SEE all 13 Source-3 batches. Since load_split() reads each split from
its own npz (common.py:48) and split_keys.json is only used to pick the batch-key
strategy, we repartition the three npz so:

    train = random 90% of all rows  -> what each method FITS on (spans all 13 batches)
    val   = the other 10%           -> completes Z_tr_corr (train + val = every row)
    test  = small random dummy       -> non-empty so projection code runs; ignored

RANDOM (not tail) partition so EVERY batch in val/test contains DMSO. TVN/sphering
transform per batch and raise if a batch has no DMSO controls; the first version's
tail slice put a DMSO-free batch-8 slice into val/test and crashed tvn/mad_tvn.
Z_tr_corr = vstack(train, val) = all rows / all 13 batches; the probe re-splits by batch.

Usage:
  python masterthesis/make_all13_split.py \
      --src checkpoints/stoic-snowball-398_9llyzmze/channelwise_embeddings \
      --dst checkpoints/stoic-snowball-398_9llyzmze/all13/channelwise_embeddings
"""
import argparse
import json
from pathlib import Path

import numpy as np

SPLITS = ["train", "validation", "test"]
FNAME = "{}_channelwise_embeddings.npz"


def load_npz(p):
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source channelwise_embeddings dir")
    ap.add_argument("--dst", required=True, help="output all-13 channelwise_embeddings dir")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of rows put in val (completes Z_tr_corr; DMSO-safe)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    dicts = [load_npz(src / FNAME.format(s)) for s in SPLITS]
    keys = list(dicts[0].keys())
    n_each = [len(d["names"]) for d in dicts]
    N = sum(n_each)

    # concatenate every per-row key; copy any non-row key from the first file
    full = {}
    for k in keys:
        if all(d[k].shape[0] == n_each[i] for i, d in enumerate(dicts)):
            full[k] = np.concatenate([d[k] for d in dicts], axis=0)
        else:
            full[k] = dicts[0][k]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    test_idx = perm[:min(4096, N)]  # dummy; projected then ignored

    def take(idx):
        return {k: (v[idx] if getattr(v, "shape", [0])[0] == N else v) for k, v in full.items()}

    np.savez(dst / FNAME.format("train"), **take(train_idx))
    np.savez(dst / FNAME.format("validation"), **take(val_idx))
    np.savez(dst / FNAME.format("test"), **take(test_idx))

    # split_keys.json one level up -> detect_batch_keys() picks split_strategy
    strat = {"split_strategy": "batches", "random_state": 42}
    src_keys = src.parent / "split_keys.json"
    if src_keys.exists():
        with open(src_keys) as f:
            info = json.load(f)
        strat = {"split_strategy": info.get("split_strategy", "batches"),
                 "random_state": info.get("random_state", 42)}
    with open(dst.parent / "split_keys.json", "w") as f:
        json.dump(strat, f, indent=2)

    print(f"train={len(train_idx)}  val={len(val_idx)}  test(dummy)={len(test_idx)}  "
          f"(total {N}) -> {dst}")
    print(f"wrote {dst.parent / 'split_keys.json'}: {strat}")


if __name__ == "__main__":
    main()
