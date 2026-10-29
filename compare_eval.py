#!/usr/bin/env python3
"""Side-by-side table of the thesis metrics in two eval_all_wellagg.json files.

usage: compare_eval.py NEW.json REFERENCE.json
"""
import json
import sys

COLS = [
    ("iLISI", ("lisi", "lisi_batch")),
    ("cLISI", ("lisi", "lisi_label_name")),
    ("kBET rej", ("kbet", "kbet_batch_rejection")),
    ("batch sil", ("silhouette", "silhouette_batch")),
    ("NMI", ("leiden", "leiden_nmi")),
    ("AUROC xbatch", ("auroc_xbatch", "auroc_replicating")),
    ("AUROC", ("auroc", "auroc_replicating")),
    ("mAP", ("map", "mean_ap")),
]
BRANCHES = ["raw", "tvn", "harmony", "combat", "scvi", "mnn", "fastmnn",
            "sphering_none", "mad_none", "seurat"]


def get(d, branch, path):
    try:
        v = d[branch]
        for k in path:
            v = v[k]
        return float(v)
    except (KeyError, TypeError, ValueError):
        return None


def main():
    new, ref = (json.load(open(p)) for p in sys.argv[1:3])
    branches = [b for b in BRANCHES if b in new or b in ref]
    print(f"{'branch':14s}" + "".join(f"{c:>24s}" for c, _ in COLS))
    print(f"{'':14s}" + "".join(f"{'new':>8s}{'thesis':>8s}{'diff':>8s}" for _ in COLS))
    for b in branches:
        row = f"{b:14s}"
        for _, p in COLS:
            a, r = get(new, b, p), get(ref, b, p)
            row += f"{a:8.3f}" if a is not None else f"{'-':>8s}"
            row += f"{r:8.3f}" if r is not None else f"{'-':>8s}"
            row += f"{a - r:+8.3f}" if (a is not None and r is not None) else " " * 8
        print(row)


if __name__ == "__main__":
    main()
