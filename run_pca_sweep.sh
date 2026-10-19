#!/usr/bin/env bash
# PCA dimension sweep: is the 50-component projection driving the ranking?
# Re-fits the five ranking-relevant methods at 50/100/200 components in an
# isolated sweep dir (symlinked npz; the canonical eval_all.json is never
# touched), evaluates all branches in one pass, then adds the cross-batch
# DMSO-excluded AUROC.
#
# Usage: ./run_pca_sweep.sh  (from the repo root that holds checkpoints/)
set -u

DIMS=(50 100 200)
METHODS=(none tvn harmony combat mnn)
META=source_3_w_neg.pq

run_sweep () {
    local RUN="$1"; local NAME="$2"
    local CE="$RUN/channelwise_embeddings"
    local SP="$RUN/pca_sweep"
    local SWEEP="$SP/channelwise_embeddings"

    if [ ! -d "$CE" ]; then echo "[skip] $NAME: no channelwise_embeddings ($CE)"; return; fi
    mkdir -p "$SWEEP"
    # ABSOLUTE symlink targets (realpath): a relative -s target resolves against
    # the link's own dir, not cwd, so relative links here would dangle.
    for f in "$CE"/*_channelwise_embeddings.npz; do ln -sf "$(realpath "$f")" "$SWEEP/"; done
    cp -n "$(realpath "$RUN/split_keys.json")" "$SP/split_keys.json"   # detect_batch_keys reads it one level up

    for d in "${DIMS[@]}"; do
        for m in "${METHODS[@]}"; do
            local sub="${m}_pca${d}"
            if [ -f "$SWEEP/$sub/Z_tr_corr.npy" ]; then
                echo "[skip] $NAME/$sub already exists"; continue
            fi
            echo "[$(date)] $NAME :: $m @ ${d}D -> $sub"
            python masterthesis/normalize_pipeline.py \
                --input-dir "$SWEEP" --meta-path "$META" \
                --method "$m" --n-components "$d" --out-subdir "$sub" \
                || echo "[FAIL] $NAME/$sub (continuing)"
        done
    done

    echo "[$(date)] $NAME :: eval_all over sweep branches"
    # (--bio-recall omitted: bio-recall is not used in the thesis and its
    # loaders are not part of this folder)
    python masterthesis/eval_all.py \
        --embeddings-dir "$SWEEP" --meta-path "$META" \
        --no-raw \
        --out-json "$SWEEP/eval_sweep.json" \
        || echo "[FAIL] eval_all $NAME (continuing)"

    echo "[$(date)] $NAME :: cross-batch AUROC variants into eval_sweep.json"
    python masterthesis/recompute_auroc_variants.py \
        --runs "$SWEEP" --json-name eval_sweep.json --meta-path "$META" \
        || echo "[FAIL] recompute_auroc_variants $NAME (continuing)"
    echo "[$(date)] $NAME :: done -> $SWEEP/eval_sweep.json"
}

echo "[$(date)] PCA dim sweep start  host=$(hostname)"
run_sweep checkpoints/stoic-snowball-398_9llyzmze ResNet
run_sweep checkpoints/honest-snowflake-408_x79nwnqn MAE
echo "[$(date)] PCA dim sweep done"
