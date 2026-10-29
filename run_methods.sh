#!/bin/bash -l
# run_methods.sh — Run all batch correction methods on channelwise embeddings
#
# Usage:
#   ./run_methods.sh --input-dir checkpoints/<run>/channelwise_embeddings [--meta-path source_3_w_neg.pq]
#
# Features:
#   - Runs the batch correction methods: tvn, harmony, combat, scvi, mnn, fastmnn
#     (seurat, sphering and mad are run separately, see README.md)
#   - Skips methods that already have output (checks for Z_tr_corr.npy)
#   - Runs the well-aggregated evaluation at the end with the settings of the thesis
#     (Leiden resolution 1.0, k = 15) and writes eval_all_wellagg.json
#   - Color-coded output for easy reading

set -e

# Environment setup (HPC conda activation; skipped when conda is already on the PATH)
if ! command -v conda >/dev/null 2>&1; then
    source ~/data/miniconda3/etc/profile.d/conda.sh
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
META_PATH="source_3_w_neg.pq"
INPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --meta-path)
            META_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --input-dir <path> [--meta-path <path>]"
            echo ""
            echo "Options:"
            echo "  --input-dir   Directory containing channelwise embeddings (required)"
            echo "  --meta-path   Path to metadata parquet file (default: source_3_w_neg.pq)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate input
if [[ -z "$INPUT_DIR" ]]; then
    echo -e "${RED}Error: --input-dir is required${NC}"
    echo "Usage: $0 --input-dir <path> [--meta-path <path>]"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}Error: Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

if [[ ! -f "$META_PATH" ]]; then
    echo -e "${RED}Error: Metadata file does not exist: $META_PATH${NC}"
    exit 1
fi

# Activate conda environment (override with CONDA_ENV=mm_cpu for a CPU-only run)
conda activate "${CONDA_ENV:-mm_cuda}"

# Run from the repository root (the directory of this script)
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=$PWD:$PYTHONPATH

echo -e "${BLUE}=========================================="
echo "Batch Correction Pipeline"
echo "=========================================="
echo -e "Input:    ${INPUT_DIR}"
echo -e "Metadata: ${META_PATH}"
echo -e "==========================================${NC}\n"

# Define all methods and their parameters
declare -A METHODS
METHODS["tvn"]=""
METHODS["harmony"]=""
METHODS["combat"]=""
METHODS["scvi"]="--n-latent 50 --n-hidden 256 --n-layers 2 --max-epochs 200"
METHODS["mnn"]="--k 20 --sigma 15.0 --alpha 0.1"
METHODS["fastmnn"]="--fastmnn-k 20 --fastmnn-sigma 1.0"

# Track counts
SKIPPED=0
RUN=0
FAILED=0

# Run each method
for method in tvn harmony combat scvi mnn fastmnn; do
    OUTPUT_DIR="${INPUT_DIR}/${method}"
    OUTPUT_FILE="${OUTPUT_DIR}/Z_tr_corr.npy"
    
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo -e "${YELLOW}[SKIP]${NC} ${method} — already exists at ${OUTPUT_DIR}"
        ((SKIPPED++)) || true
    else
        echo -e "${BLUE}[RUN]${NC} ${method}..."
        
        EXTRA_ARGS="${METHODS[$method]}"
        
        if python masterthesis/normalize_pipeline.py \
            --input-dir "$INPUT_DIR" \
            --meta-path "$META_PATH" \
            --method "$method" \
            $EXTRA_ARGS; then
            echo -e "${GREEN}[DONE]${NC} ${method} -> ${OUTPUT_DIR}\n"
            ((RUN++)) || true
        else
            echo -e "${RED}[FAIL]${NC} ${method} failed!\n"
            ((FAILED++)) || true
        fi
    fi
done

# Summary
echo -e "${BLUE}=========================================="
echo "Method Summary"
echo -e "==========================================${NC}"
echo -e "  Skipped (already exist): ${SKIPPED}"
echo -e "  Run successfully:        ${RUN}"
echo -e "  Failed:                  ${FAILED}"

# Run evaluation if any methods were run or if eval_all_wellagg.json doesn't exist.
# Settings of the thesis: sites pooled per well, Leiden resolution 1.0 (k = 15).
EVAL_JSON="${INPUT_DIR}/eval_all_wellagg.json"
if [[ $RUN -gt 0 ]] || [[ ! -f "$EVAL_JSON" ]]; then
    echo -e "\n${BLUE}=========================================="
    echo "Running Comprehensive Evaluation"
    echo -e "==========================================${NC}"

    python masterthesis/eval_all.py \
        --embeddings-dir "$INPUT_DIR" \
        --meta-path "$META_PATH" \
        --aggregate-level well \
        --leiden-resolution 1.0 \
        --out-json "$EVAL_JSON"
    
    echo -e "\n${GREEN}Evaluation complete!${NC}"
    echo -e "Results saved to: ${EVAL_JSON}"
else
    echo -e "\n${YELLOW}Skipping evaluation — no new methods run and eval_all.json exists${NC}"
fi

echo -e "\n${GREEN}=========================================="
echo "Pipeline Complete!"
echo -e "==========================================${NC}"
