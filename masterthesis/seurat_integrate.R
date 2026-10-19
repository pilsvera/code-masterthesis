#!/usr/bin/env Rscript
# =============================================================================
# seurat_integrate.R  --  Seurat v5 reciprocal-PCA (RPCA) batch integration of a
# pre-computed low-dimensional embedding.
#
# Called by seurat_method.py (the Python bridge). Not meant to be run by hand.
#
# Args:
#   1  emb_in.csv    N x K numeric matrix (cells x embedding dims), NO header
#   2  batch_in.csv  one column 'batch', N rows (the group to integrate over)
#   3  emb_out.csv   OUTPUT: N x D corrected embedding, NO header (row order == input)
#   4  dims          integer, #dims for the RPCA reduction (default 30, scib default)
#
# Design note: the thesis pipeline corrects the 50-PC train+val space (harmony.py
# does the same), so this integrates that same PCA matrix rather than the raw
# features. The PCs are treated as the assay 'data' directly (they are already a
# continuous embedding -- no LogNormalize). Output is the RPCA-corrected reduction
# (N x dims), cell order preserved.
# =============================================================================
suppressMessages({
  library(Seurat)
  library(SeuratObject)
  library(future)
})
# Seurat's IntegrateLayers ships the split data to future workers; the default
# 500 MiB globals guard rejects the ~1.5 GiB integration payload. Lift it (we have
# the RAM) and keep the plan sequential so nothing is copied across processes.
plan(sequential)
options(future.globals.maxSize = 16 * 1024^3)

args   <- commandArgs(trailingOnly = TRUE)
emb_in <- args[[1]]
bat_in <- args[[2]]
emb_out <- args[[3]]
dims_n <- if (length(args) >= 4) as.integer(args[[4]]) else 30L

# --- fast numeric read (scan is much faster than read.csv on ~2e5 rows) ---
K   <- length(strsplit(readLines(emb_in, n = 1), ",")[[1]])
vec <- scan(emb_in, sep = ",", quiet = TRUE)
emb <- matrix(vec, ncol = K, byrow = TRUE)          # N x K
storage.mode(emb) <- "double"
batch <- as.character(read.csv(bat_in, stringsAsFactors = FALSE)$batch)

N <- nrow(emb)
nb <- length(unique(batch))
cat(sprintf("[R] %d cells x %d dims, %d '%s'-groups\n", N, K, nb, "batch"))
stopifnot(length(batch) == N)

# features x cells; name so we can restore original order after integration
mat <- t(emb)
rownames(mat) <- paste0("PC", seq_len(K))
colnames(mat) <- paste0("cell", seq_len(N))

# Build a v5 assay straight from the embedding as the 'data' layer (no counts, so
# negative PC values are fine and no normalization is implied).
assay <- CreateAssay5Object(data = mat)
obj   <- CreateSeuratObject(assay)
obj$batch <- batch
VariableFeatures(obj) <- rownames(obj)               # all K PCs are the features

# split into per-group layers -> RPCA integrates across them
obj[["RNA"]] <- split(obj[["RNA"]], f = obj$batch)

npcs <- min(dims_n, K - 1L)
smallest <- min(table(batch))
kw <- max(10L, min(100L, smallest - 1L))             # k.weight must be < smallest group
cat(sprintf("[R] npcs=%d  k.weight=%d  (smallest group=%d)\n", npcs, kw, smallest))

obj <- ScaleData(obj, features = rownames(obj), verbose = FALSE)
obj <- RunPCA(obj, features = rownames(obj), npcs = npcs, approx = FALSE, verbose = FALSE)
obj <- IntegrateLayers(
  obj, method = RPCAIntegration,
  orig.reduction = "pca", new.reduction = "integrated.rpca",
  dims = 1:npcs, k.weight = kw, verbose = FALSE
)

Z <- Embeddings(obj, "integrated.rpca")              # N x npcs
Z <- Z[colnames(mat), , drop = FALSE]                # restore original cell order
write.table(Z, emb_out, sep = ",", row.names = FALSE, col.names = FALSE)
cat(sprintf("[R] wrote corrected %d x %d -> %s\n", nrow(Z), ncol(Z), emb_out))
