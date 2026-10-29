# code-masterthesis

Code, metadata and pinned environments that produced every result of the
master's thesis on batch correction of learned Cell Painting embeddings
(Vera Pils, Johannes Kepler University Linz, 2026). The thesis trains two
encoders (ResNet50_Modified, OpenPhenom-MAE) on JUMP-CP images, applies
nine batch-correction methods to the learned embeddings and scores batch
mixing and biological preservation on the well level.

## Scope of the repository

This repository contains the code required to reproduce the thesis
results, the canonical metadata files and the two environment
specifications. Every included module is either a result-producing entry
point or part of the import chain of one (verified against the provenance
appendix of the thesis and the import graph).

The following are deliberately excluded:

- the JUMP-CP images (public, see *Data*) and the trained checkpoints and
  embeddings (large; available on request),
- the SLURM wrappers of the cluster (site-specific; each of them calls the
  entry points below with the arguments given in this README), with one
  exception: `gpu_job_matrix_resnet.slrm`, the array job that trained the
  twelve ResNet runs, is included because it records the exact training
  arguments of the thesis,
- the figure- and table-drawing scripts (the provenance appendix of the
  thesis lists the input run and script for every figure and table),
- modules that the thesis results never used (Grad-CAM, BBKNN, DESC and the
  pathway-recall metrics are not included).

## Data

### Images

All images come from the JUMP-CP principal dataset `cpg0016`
(Chandrasekaran et al. 2023), which is public on the Cell Painting
Gallery (AWS S3, no credentials needed):

- Source 3, the primary dataset: `s3://cellpainting-gallery/cpg0016-jump/source_3/`
  (13 batches, 303 plates, 282,951 site images of the eight positive-control
  compounds and DMSO),
- Source 1, the second laboratory: `s3://cellpainting-gallery/cpg0016-jump/source_1/`
  (6 batches, 71,940 site images, same compounds).

```bash
aws s3 ls --no-sign-request s3://cellpainting-gallery/cpg0016-jump/source_3/
```

The metadata parquets list, for every site, the five original channel
files and their paths (`FileName_Orig*`, `PathName_Orig*` and `AWS_Path*`
columns), so the exact subset the thesis used can be downloaded from
those columns. The training loader (`dataset.py`) expects the five
channels of a site stacked into one 8-bit TIFF file named
`<Metadata_Sample_ID>.jpg` (TIFF content, the `.jpg` extension is
historical; the file is read with `tifffile`), channel order Mito, AGP,
RNA, ER, DNA, all files in one directory given by `image_path` in the
training config. The stacking was done once before training and is not
part of the pipeline.

### Metadata

This folder carries the canonical metadata used for every thesis result:
`source_3_w_neg.pq` (282,951 rows) and `source_1_w_neg.pq` (the image-fix
version, 71,940 rows). The `*_encoding_w_neg_*.json` files document the
label maps; they match the `.pq` columns exactly. `train.py` reads
`<source_types>.pq` directly; its `--index all_sources.pq` fallback (an index
builder) is only consulted when that parquet is absent. The code reads the label
maps from the `.pq` columns (`pert_iname`/`pert_iname_str`), never from
the JSONs, which `dataset.py` only writes. Source-3 mapping: AMG900=0,
DMSO=1, FK-866=2, LY2109761=3, NVS-PAK1-1=4, TC-S-7004=5, aloxistatin=6,
dexamethasone=7, quinidine=8; batches CP59=0 ... CP_36=12. Source 1 uses
the identical compound mapping with batches Batch1_20221004=0 ...
Batch6_20221102=5.

## Environment

- Local work (correction, evaluation, analyses): `conda env create -f mm_cpu.yaml`
  (Python 3.10, CPU PyTorch, scanpy, harmonypy, scvi-tools, leidenalg, copairs).
- Cluster (every number in the thesis): `mm_cuda_cluster_env.txt` is the
  pinned package list of the conda environment actually used (Python
  3.10.18, PyTorch 2.5.1 with CUDA 12.1, numpy 2.2.6, scanpy 1.11.5,
  scvi-tools 1.3.3, Scanorama 1.7.4, harmonypy, leidenalg, copairs).
  Training and scVI need a GPU.
- Seurat: `seurat_method.py` calls `seurat_integrate.R`, which needs an R
  environment with Seurat v5.
- Sphering: `sphering.py` imports pycytominer's `Spherize` from a clone of
  the Arevalo et al. (2024) repository
  (https://github.com/carpenter-singh-lab/2023_Arevalo_NatComm_BatchCorrection) placed as
  `2023_Arevalo_NatComm_BatchCorrection/` next to `masterthesis/`, or from
  an installed `pycytominer`.
- Run everything from the repository root with `export PYTHONPATH=$PWD:$PYTHONPATH`.

## Reproducing the thesis numbers

On a SLURM cluster the canonical ResNet cell of the thesis (batches split, seed 42)
runs end to end with one command, which chains the four steps below:

```bash
IMAGES=/path/to/source_3 ./reproduce_canonical.sh
```

### 1. Train an encoder and extract the embeddings

```bash
python masterthesis/train.py --config masterthesis/config/wandb_config.yaml
```

The twelve ResNet runs of the thesis (four split strategies, seeds 12, 24
and 42) were trained with the SLURM array job `gpu_job_matrix_resnet.slrm`,
which passes every training argument explicitly and takes the split and the
seed from the array index (the mapping is listed in the script):

```bash
sbatch gpu_job_matrix_resnet.slrm   # array 0-11: 4 splits x seeds 12, 24, 42
```

Each cell of the thesis ran into the job time limit rather than into early stopping,
after 13 to 15 of the 100 configured epochs, and the benchmark used the checkpoint with
the best validation accuracy. The limit in the script is now two days, which is enough
for the configured 100 epochs at about 16 minutes per epoch.

Submit it from the repository root (it changes into `SLURM_SUBMIT_DIR`), with
the images at `masterthesis/source_3/`. Normalisation constants: the thesis
lists the per-channel means and standard deviations applied to the Source-3
images (Appendix F). In `custom_meanstd.py` these constants are stored under
the name `CELLPAINTING_1`; the original runs were launched with `--mean-std
CELLPAINTING_3_w_NEG`, but `train.py` at that time ignored the flag and applied
`CELLPAINTING_1`. The shipped `train.py` honours the flag, so the array job
names the constants that were actually used.

The config sets the encoder, the metadata file (`source_types`), the image
directory (`image_path`), the split strategy and the seed. Training logs to
Weights & Biases (`WANDB_MODE=offline` keeps the logs local). Each run writes
`checkpoints/<run-name>_<wandb-id>/` with the checkpoint, the config,
`split_keys.json` (split strategy and seed, the source of truth) and
`channelwise_embeddings/{train,validation,test}_channelwise_embeddings.npz`.
The canonical runs of the thesis are `stoic-snowball-398_9llyzmze`
(ResNet50_Modified) and `honest-snowflake-408_x79nwnqn` (OpenPhenom-MAE),
both batches split, seed 42; the seed panels use seeds 12, 24 and 42.
Offline runs are named by their wandb id.

#### Extracting the embeddings

Training alone writes the checkpoint (`ResNet50_Modified_best_model.pth`, the
best validation epoch, plus `*_epoch_N.pth` every fifth epoch). The channelwise
embeddings come from a second call of `train.py` that resumes the checkpoint
and, because of `--return-channelwise-embeddings true`, only passes the three
splits through the network:

```bash
R=checkpoints/<run-name>_<wandb-id>
python masterthesis/train.py --architecture ResNet50_Modified --source-types source_3_w_neg \
    --mean-std CELLPAINTING_1 --image-path masterthesis/source_3/ --resize 224 224 \
    --eval-split 0.2 --test-split 0.1 --batch-size 64 --num-workers 8 --pretrained --no-freeze \
    --splits batches --seed 42 \
    --resume --checkpoint $R/ResNet50_Modified_best_model.pth --return-channelwise-embeddings true
```

`--splits` and `--seed` repeat the values of the run (`split_keys.json`).
Without `--config`, the call reads the augmentation block of
`masterthesis/config/wandb_config.yaml`: the train split is dumped through that
augmentation (random resized crop, flips, rotation, noise, blur,
brightness/contrast, coarse dropout, plasma shadow), the validation and test
splits without augmentation. This is how the thesis embeddings were produced.
When resuming, `train.py` applies the `CELLPAINTING_1` constants regardless of
`--mean-std`, which every thesis checkpoint was trained with. Output:
`$R/channelwise_embeddings/{train,validation,test}_channelwise_embeddings.npz`.

#### OpenPhenom-MAE and Source 1

The OpenPhenom-MAE embeddings come from the published encoder. When `train.py`
resumes an OpenPhenom-MAE run it loads only the classification head from the
checkpoint and keeps the encoder weights from Hugging Face
(`recursionpharma/OpenPhenom`), so the MAE training runs only provide the run
directory and the split.

The Source-1 replication (second laboratory) uses the Source-1 metadata,
images and normalisation constants. These are the commands of the thesis runs
`eager-snowball-414` (ResNet) and `generous-pond-415` (MAE):

```bash
# ResNet: training, then extraction from the epoch-5 checkpoint of that run
python masterthesis/train.py --architecture ResNet50_Modified --source-types source_1_w_neg \
    --mean-std CELLPAINTING_1_w_NEG --image-path masterthesis/source_1/ \
    --batch-size 256 --epochs 100 --lr 0.001 --lr-scheduler auto --pretrained --no-freeze \
    --early-stopping-metric Accuracy --early-stopping-patience 5 --eval-split 0.2 --test-split 0.1 \
    --resize 224 224 --horizontal-flip-prob 0.5 --vertical-flip-prob 0.5 --rotation-prob 0 \
    --noise-std 0.2 --noise-prob 0.5 --brightness-contrast-prob 0.5 --blur-prob 0 \
    --coarse-dropout-prob 0 --return-channelwise-embeddings false --num-workers 16 \
    --splits batches --seed 42
python masterthesis/train.py --config masterthesis/config/extract_resnet_s1_wneg.yaml \
    --architecture ResNet50_Modified --source-types source_1_w_neg --mean-std CELLPAINTING_1_w_NEG \
    --image-path masterthesis/source_1/ --resize 224 224 \
    --eval-split 0.2 --test-split 0.1 --batch-size 64 --num-workers 8 --pretrained --no-freeze \
    --splits batches --seed 42 --resume --checkpoint $R/<timestamp>_ResNet50_Modified_epoch_5.pth \
    --return-channelwise-embeddings true

# MAE: training run, then extraction with augmentation off
python masterthesis/train.py --config masterthesis/config/wandb_config_no_aug.yaml \
    --architecture OpenPhenomMAE --source-types source_1_w_neg \
    --mean-std CELLPAINTING_1_w_NEG --image-path masterthesis/source_1/ \
    --batch-size 64 --epochs 100 --lr 0.001 --lr-scheduler auto --pretrained --no-freeze \
    --early-stopping-metric Accuracy --early-stopping-patience 5 --eval-split 0.2 --test-split 0.1 \
    --resize 256 256 --horizontal-flip-prob 0.5 --vertical-flip-prob 0.5 --rotation-prob 0 \
    --noise-std 0.2 --noise-prob 0.5 --brightness-contrast-prob 0.5 --blur-prob 0 \
    --coarse-dropout-prob 0 --return-channelwise-embeddings false --num-workers 8 \
    --splits batches --seed 42
python masterthesis/train.py --config masterthesis/config/extract_mae_s1_wneg.yaml \
    --architecture OpenPhenomMAE --source-types source_1_w_neg --mean-std CELLPAINTING_1_w_NEG \
    --image-path masterthesis/source_1/ --resize 256 256 \
    --eval-split 0.2 --test-split 0.1 --batch-size 64 --num-workers 8 --pretrained --no-freeze \
    --horizontal-flip-prob 0 --vertical-flip-prob 0 --rotation-prob 0 --noise-prob 0 \
    --brightness-contrast-prob 0 --blur-prob 0 --coarse-dropout-prob 0 \
    --splits batches --seed 42 --resume --checkpoint $R/OpenPhenomMAE_best_model.pth \
    --return-channelwise-embeddings true
```

The two `extract_*_s1_wneg.yaml` configs carry `CELLPAINTING_1_w_NEG` inside
the augmentation block. A resumed run otherwise falls back to `CELLPAINTING_1`,
which matches the Source-3 checkpoints but not these two runs.

### 2. Apply the corrections

```bash
./run_methods.sh --input-dir checkpoints/<run>/channelwise_embeddings --meta-path source_3_w_neg.pq
```

runs TVN, Harmony, ComBat, scVI, MNN (Scanorama) and fastMNN, skips
branches that already exist, and ends with the evaluation of step 3. The
remaining branches of the thesis are single calls of the same CLI:

```bash
D=checkpoints/<run>/channelwise_embeddings
python masterthesis/normalize_pipeline.py --input-dir $D --meta-path source_3_w_neg.pq --method seurat
python masterthesis/normalize_pipeline.py --input-dir $D --meta-path source_3_w_neg.pq --pre sphering --method none
python masterthesis/normalize_pipeline.py --input-dir $D --meta-path source_3_w_neg.pq --pre mad --method none
```

The preprocessing combinations of the appendix tables put `--pre mad` or
`--pre sphering` in front of any `--method`. Every branch is a
subdirectory of `channelwise_embeddings/` named after the combination
(`harmony/`, `sphering_none/`, `mad_harmony/`, ...) holding `Z_tr_corr.npy`,
`meta_tr.pq`, `Z_te_corr.npy`, `meta_te.pq` and `pca.pkl` (50 principal
components, scaler and PCA fitted on the train split only).

### 3. Score every branch

The thesis scores the well-aggregated embeddings with Leiden resolution 1.0 and k = 15,
which `run_methods.sh` passes explicitly and `eval_all.py` now uses by default.

```bash
python masterthesis/eval_all.py --embeddings-dir $D --meta-path source_3_w_neg.pq \
    --aggregate-level well --leiden-resolution 1.0 \
    --out-json $D/eval_all_wellagg.json
python masterthesis/recompute_auroc_variants.py --runs $D
```

`eval_all.py` discovers every branch directory, pools the sites of each
well to one profile and computes all metrics of the thesis on the
train+validation wells. The thesis settings are well aggregation and Leiden
resolution 1.0 with k = 15, which are the argument
defaults. `recompute_auroc_variants.py` then adds the cross-batch,
DMSO-excluded replicate AUROC, the primary biological metric of the thesis,
to the same JSON. `run_methods.sh` already ends with the first command.

### 4. Deployment and control analyses

| Thesis result | Command |
|---|---|
| Leave-one-batch-out of the deployable corrections (TVN, Sphering, MAD) | `python masterthesis/figures/tvn_lobo.py --ce $D --meta-path source_3_w_neg.pq --out-dir <out>` writes `tvn_lobo_per_batch.csv` (both probe variants, `bacc` nine-way and `bacc_xdmso` eight-way; the thesis reports `bacc_xdmso`); `python masterthesis/figures/lobo_live_batches.py` restricts the summary to the batches with compound signal |
| Source-3-to-Source-1 transfer | `python masterthesis/figures/cross_source_deploy.py --encoder <name> --ref-ce <source-3 dir> --deploy-ce <source-1 dir> --ref-meta source_3_w_neg.pq --deploy-meta source_1_w_neg.pq --out-dir <out>` |
| Integration probe (all 13 batches corrected jointly, linear head) | `python masterthesis/make_all13_split.py --src <run> --dst <run_all13>`, corrections as in step 2, then `python masterthesis/probe_integration.py --input-dir <run_all13>/channelwise_embeddings` |
| Batch quality control (within- and cross-batch probes per batch) | `python masterthesis/deadbatch_probes.py --runs <run dirs> --out-dir <out>` writes `deadbatch_diagnosis.json` |
| Nested variance and plate-position control | `python masterthesis/plate_effect.py --encoders <names> --out-dir <out>` writes `plate_effect.json` |
| Source/batch/plate/well variance decomposition | `python masterthesis/four_level_decomp.py` (expects `checkpoints/` next to `masterthesis/`, or `THESIS_REPO=<root>`) |
| Projection-dimension sweep (50/100/200 components) | `./run_pca_sweep.sh` from the directory that holds `checkpoints/` |

## Layout

- `masterthesis/` is the Python package (run with `PYTHONPATH=$PWD`).
  - Training: `train.py`, `model.py`, `vit.py`, `mae_modules.py`,
    `mae_utils.py`, `masking.py`, `huggingface_mae.py`, `dataset.py`,
    `loss.py`, `normalizer.py`, `custom_meanstd.py`, `mm_logging.py`,
    `config/` (wandb YAMLs and validator).
  - Batch correction: `normalize_pipeline.py` (single CLI) plus one module
    per method: `tvn.py`, `harmony.py`, `combat.py`, `mad.py`,
    `sphering.py`, `mnn.py` (Scanorama), `fastmnn.py`, `scvi_method.py`,
    `seurat_method.py` with `seurat_integrate.R`. Shared helpers in
    `common.py`.
  - Evaluation: `eval_all.py`, `metrics.py`, `recompute_auroc_variants.py`.
  - Deployment and control analyses: the scripts of step 4.
- `run_methods.sh`, `run_pca_sweep.sh`: the two shell drivers.
- `reproduce_canonical.sh`, `reproduce_post.slrm`, `compare_eval.py`: the one-command
  reproduction of the canonical cell and the side-by-side metric table against a
  reference `eval_all_wellagg.json`.
- `gpu_job_matrix_resnet.slrm`: the SLURM array job of the twelve ResNet
  training runs.
- `mm_cpu.yaml`, `mm_cuda_cluster_env.txt`: the two environments.
- `source_3_w_neg.pq`, `source_1_w_neg.pq`, `*_encoding_w_neg_*.json`: the
  metadata.
