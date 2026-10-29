#!/bin/bash
# Reproduce the canonical ResNet cell of the thesis (batches split, seed 42) from this
# repository, following README.md. Two chained SLURM jobs are submitted:
#   1. gpu_job_matrix_resnet.slrm, array index 2   -> training (README step 1)
#   2. reproduce_post.slrm, after job 1 has ended  -> extraction, corrections, scoring
#
# The only step outside the repository is the data placement: the stacked five-channel
# site images have to be at masterthesis/source_3/ (README, Data). Pass IMAGES=<dir> to
# symlink them from elsewhere. Sphering needs the Arevalo et al. repository next to
# masterthesis/ (README, Environment); it is cloned here if it is absent.
set -eo pipefail
cd "$(dirname "$0")"

if [ -n "${IMAGES:-}" ] && [ ! -e masterthesis/source_3 ]; then
    ln -s "$IMAGES" masterthesis/source_3
fi
if [ ! -e masterthesis/source_3 ]; then
    echo "masterthesis/source_3 is missing. Place the stacked site images there, or run"
    echo "    IMAGES=/path/to/source_3 $0"
    exit 1
fi
[ -d 2023_Arevalo_NatComm_BatchCorrection ] || git clone -q --depth 1 \
    https://github.com/carpenter-singh-lab/2023_Arevalo_NatComm_BatchCorrection

export WANDB_MODE=offline

j1=$(sbatch --parsable --export=ALL --array=2 gpu_job_matrix_resnet.slrm)
echo "training job $j1 (array index 2 = batches split, seed 42)"
j2=$(sbatch --parsable --export=ALL --dependency=afterany:"$j1" reproduce_post.slrm)
echo "post job $j2 (extraction, corrections, scoring; starts once $j1 has ended)"
echo
echo "Watch:   squeue -u \$USER   and   tail -f slurm-${j1}_2.out"
echo "Result:  checkpoints/<run>/channelwise_embeddings/eval_all_wellagg.json"
echo "Set REF_JSON=<thesis eval_all_wellagg.json> for the side-by-side comparison."
