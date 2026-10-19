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
  entry points below with the arguments given in this README),
- the figure- and table-drawing scripts (the provenance appendix of the
  thesis lists the input run and script for every figure and table),
- modules that the thesis results never used (`bbknn_method.py` is kept
  only because `run_methods.sh` calls it; BBKNN is not part of the thesis).

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
label maps; they match the `.pq` columns exactly. The code reads the label
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
  the Arevalo et al. (2024) repository placed as
  `2023_Arevalo_NatComm_BatchCorrection/` next to `masterthesis/`, or from
  an installed `pycytominer`.
- Run everything from the repository root with `export PYTHONPATH=$PWD:$PYTHONPATH`.

## Reproducing the thesis numbers

### 1. Train an encoder and extract the embeddings

```bash
python masterthesis/train.py --config masterthesis/config/wandb_config.yaml
```

The config sets the encoder, the metadata file (`source_types`), the image
directory (`image_path`), the split strategy and the seed. Training logs to
Weights & Biases (`WANDB_MODE=offline` keeps the logs local). Each run writes
`checkpoints/<run-name>_<wandb-id>/` with the checkpoint, the config,
`split_keys.json` (split strategy and seed, the source of truth) and
`channelwise_embeddings/{train,validation,test}_channelwise_embeddings.npz`.
The canonical runs of the thesis are `stoic-snowball-398_9llyzmze`
(ResNet50_Modified) and `honest-snowflake-408_x79nwnqn` (OpenPhenom-MAE),
both batches split, seed 42; the seed panels use seeds 12, 24 and 42.

### 2. Apply the corrections

```bash
./run_methods.sh --input-dir checkpoints/<run>/channelwise_embeddings --meta-path source_3_w_neg.pq
```

runs TVN, Harmony, ComBat, scVI, MNN (Scanorama), fastMNN and BBKNN, skips
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

```bash
python masterthesis/eval_all.py --embeddings-dir $D --meta-path source_3_w_neg.pq \
    --aggregate-level well --leiden-resolution 1.0 --skip-branches bbknn \
    --out-json $D/eval_all_wellagg.json
python masterthesis/recompute_auroc_variants.py --runs $D
```

`eval_all.py` discovers every branch directory, pools the sites of each
well to one profile and computes all metrics of the thesis on the
train+validation wells. The thesis settings are well aggregation and Leiden
resolution 1.0 with k = 15 (the argument default of 9.0 is not what the
thesis reports). `recompute_auroc_variants.py` then adds the cross-batch,
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
| Activity-mAP variant of the metric appendix | `python masterthesis/figures/check_treatment_vs_dmso.py --out <csv>` |

## Layout

- `masterthesis/` is the Python package (run with `PYTHONPATH=$PWD`).
  - Training: `train.py`, `model.py`, `vit.py`, `mae_modules.py`,
    `mae_utils.py`, `masking.py`, `huggingface_mae.py`, `dataset.py`,
    `loss.py`, `normalizer.py`, `custom_meanstd.py`, `mm_logging.py`,
    `config/` (wandb YAMLs and validator). `gradcam.py` and `topK.py` are
    `train.py` imports and only for that reason here.
  - Batch correction: `normalize_pipeline.py` (single CLI) plus one module
    per method: `tvn.py`, `harmony.py`, `combat.py`, `mad.py`,
    `sphering.py`, `mnn.py` (Scanorama), `fastmnn.py`, `scvi_method.py`,
    `seurat_method.py` with `seurat_integrate.R`, `desc_method.py`,
    `bbknn_method.py` (the last two are not part of the thesis). Shared
    helpers in `common.py`.
  - Evaluation: `eval_all.py`, `metrics.py`, `recompute_auroc_variants.py`.
    `eval_all.py --bio-recall` is not supported here (the thesis does not
    use the pathway-recall metrics).
  - Deployment and control analyses: the scripts of step 4.
- `run_methods.sh`, `run_pca_sweep.sh`: the two shell drivers.
- `mm_cpu.yaml`, `mm_cuda_cluster_env.txt`: the two environments.
- `source_3_w_neg.pq`, `source_1_w_neg.pq`, `*_encoding_w_neg_*.json`: the
  metadata.
