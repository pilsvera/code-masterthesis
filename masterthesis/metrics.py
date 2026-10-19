"""
metrics.py — Embedding evaluation metrics for batch correction assessment.

Functions
---------
compute_silhouette   Silhouette score per batch / label key
compute_lisi         LISI (iLISI / cLISI) per batch / label key
compute_kbet         kBET rejection rate per batch key
compute_leiden       Leiden clustering ARI / NMI vs compound labels
compute_map          Mean Average Precision for compound replicate retrieval
auroc_replicates     AUROC for replicate-vs-null pair classification
probe_balanced_accuracy  Cross-set linear-probe balanced accuracy (deployment)
evaluate_branch      Run all metrics for one embedding branch
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (silhouette_score, roc_auc_score,
                             adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2
import harmonypy as hm
import copairs.map as copairs_map


# --------------------------------------------------------------------------- #
# Internal helper
# --------------------------------------------------------------------------- #
def _l2_normalize(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norms, 1e-8)


# --------------------------------------------------------------------------- #
# Batch mixing metrics  (use subsampled embeddings)
# --------------------------------------------------------------------------- #
def compute_silhouette(Z, meta, batch_keys, label_key="label_name",
                       sample_size=5000):
    """
    Silhouette score for each batch key and for the label key.

    Ideal: silhouette(batch) → 0  (batches not separable)
           silhouette(label) → 1  (compounds well-separated)
    """
    results = {}
    for key in batch_keys + [label_key]:
        labels = meta[key].astype(str).values
        if len(np.unique(labels)) < 2:
            results[f"silhouette_{key}"] = float("nan")
            continue
        results[f"silhouette_{key}"] = silhouette_score(
            Z, labels,
            sample_size=min(sample_size, len(labels)),
            random_state=42,
        )
    return results


def compute_lisi(Z, meta, batch_keys, label_key="label_name", perplexity=30):
    """
    LISI scores (Local Inverse Simpson's Index).

    iLISI (batch keys): higher → better mixing across batches.
    cLISI (label key):  lower  → tighter per-compound clusters.
    """
    cols = batch_keys + [label_key]
    meta_lisi = meta[cols].copy().astype(str)
    lisi = hm.compute_lisi(Z, meta_lisi, cols, perplexity=perplexity)
    return {f"lisi_{col}": float(np.median(lisi[:, i]))
            for i, col in enumerate(cols)}


def compute_kbet(Z, meta, batch_key, k=50, subsample=5000, seed=42):
    """
    kBET: k-nearest-neighbour Batch Effect Test (pure Python approximation).

    Adapted from Büttner et al. (2019) Nature Methods,
    "A test metric for assessing single-cell RNA-seq batch correction".
    Reference pipeline (R/rpy2): Arevalo et al. (2024) Nature Communications.

    For each (subsampled) cell, builds a k-NN neighbourhood and runs a
    chi-squared test comparing the *observed* batch distribution to the
    *expected* (global) distribution.

        kBET rejection rate  = fraction of cells where H0 is rejected at α=0.05
        kBET acceptance rate = 1 - rejection rate

    Lower rejection rate → better batch mixing.
    """
    batch_labels = meta[batch_key].astype(str).values
    batches, batch_counts = np.unique(batch_labels, return_counts=True)
    n_batches = len(batches)

    if n_batches < 2:
        return {f"kbet_{batch_key}_rejection": float("nan"),
                f"kbet_{batch_key}_acceptance": float("nan")}

    N = Z.shape[0]
    rng = np.random.default_rng(seed)
    if subsample and N > subsample:
        idx   = rng.choice(N, size=subsample, replace=False)
        Z_s   = Z[idx]
        bat_s = batch_labels[idx]
    else:
        Z_s   = Z
        bat_s = batch_labels

    expected_freq = batch_counts / batch_counts.sum()

    k_actual = min(k, Z_s.shape[0] - 1)
    nbrs = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="auto",
                            metric="euclidean")
    nbrs.fit(Z_s)
    _, indices = nbrs.kneighbors(Z_s)
    indices = indices[:, 1:]  # drop self

    n_rejected = 0
    df = n_batches - 1
    for i in range(len(Z_s)):
        neighbour_batches = bat_s[indices[i]]
        observed  = np.array([(neighbour_batches == b).sum() for b in batches],
                              dtype=float)
        expected  = expected_freq * k_actual
        chi2_stat = float(np.sum((observed - expected) ** 2
                                 / np.maximum(expected, 1e-8)))
        if 1.0 - chi2.cdf(chi2_stat, df) < 0.05:
            n_rejected += 1

    rejection  = round(n_rejected / len(Z_s), 4)
    acceptance = round(1.0 - rejection, 4)
    return {f"kbet_{batch_key}_rejection":  rejection,
            f"kbet_{batch_key}_acceptance": acceptance}


# --------------------------------------------------------------------------- #
# Leiden clustering  (use subsampled embeddings)
# --------------------------------------------------------------------------- #
def compute_leiden(Z, meta, label_key="label_name", k=15, resolution=1.0, seed=42):
    """
    Leiden clustering ARI and NMI vs. ground-truth compound labels.

    Builds a k-NN graph on Z, runs Leiden community detection, then compares
    the resulting partition to true compound labels.

    Higher ARI / NMI → Leiden clusters align better with compound identity
    (i.e. batch correction preserved biological structure).

    Adapted from Arevalo et al. (2024) Nature Communications,
    "Evaluating batch correction methods for image-based cell profiling".
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        print("  WARNING: leidenalg / igraph not installed — skipping Leiden metrics.")
        return {"leiden_ari": float("nan"), "leiden_nmi": float("nan"),
                "leiden_n_clusters": 0}

    labels = meta[label_key].astype(str).values
    N = Z.shape[0]

    # Build k-NN graph (symmetric: add both directions)
    k_actual = min(k, N - 1)
    nbrs = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="auto",
                            metric="euclidean")
    nbrs.fit(Z)
    _, indices = nbrs.kneighbors(Z)
    indices = indices[:, 1:]   # drop self

    edges = set()
    for i in range(N):
        for j in indices[i]:
            edges.add((min(i, int(j)), max(i, int(j))))
    g = ig.Graph(n=N, edges=list(edges))

    # Leiden clustering
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    cluster_labels = partition.membership

    ari = float(adjusted_rand_score(labels, cluster_labels))
    nmi = float(normalized_mutual_info_score(labels, cluster_labels,
                                             average_method="arithmetic"))
    return {
        "leiden_ari":        round(ari, 4),
        "leiden_nmi":        round(nmi, 4),
        "leiden_n_clusters": len(set(cluster_labels)),
    }


# --------------------------------------------------------------------------- #
# Replication quality metrics  (use full embeddings)
# --------------------------------------------------------------------------- #
def compute_map(Z, meta, label_key="label_name", control_key="DMSO",
                pos_diffby=None,
                batch_size=20000, null_size=10000, threshold=0.05, seed=42):
    """
    Mean Average Precision (mAP) for compound replicate retrieval.

    Adapted from Arevalo et al. (2024) Nature Communications,
    "Evaluating batch correction methods for image-based cell profiling".
    https://github.com/cytomining/copairs

    For each well, AP measures how well same-compound replicates rank above
    cross-compound pairs by cosine similarity. mAP is averaged per compound.
    percent_replicating = fraction of compounds with significant mAP
    (Benjamini-Hochberg corrected p-value < threshold).

    Controls are given unique IDs to prevent them from forming replicate pairs
    and are excluded from the final mAP summary.
    """
    pos_diffby = pos_diffby or []
    cols = [label_key, "plate"] + [c for c in pos_diffby if c not in (label_key, "plate")]
    meta_cp = meta[cols].astype(str).copy().reset_index(drop=True)

    is_ctrl  = meta_cp[label_key] == control_key
    ctrl_ids = [f"{control_key}_{i}" for i in range(int(is_ctrl.sum()))]
    meta_cp.loc[is_ctrl.values, label_key] = ctrl_ids

    ap_scores = copairs_map.average_precision(
        meta_cp, Z,
        pos_sameby=[label_key], pos_diffby=pos_diffby,
        neg_sameby=["plate"],   neg_diffby=[label_key],
        batch_size=batch_size,  distance="cosine",
        progress_bar=False,
    )

    ap_scores = ap_scores[~is_ctrl.values].copy()
    ap_scores = ap_scores[ap_scores["n_pos_pairs"] > 0]

    if ap_scores.empty:
        return {"mean_ap": float("nan"), "median_ap": float("nan"),
                "percent_replicating": float("nan"), "n_compounds": 0}

    map_df = copairs_map.mean_average_precision(
        ap_scores, sameby=[label_key],
        null_size=null_size, threshold=threshold, seed=seed,
        progress_bar=False,
    )

    return {
        "mean_ap":             round(float(map_df["mean_average_precision"].mean()), 4),
        "median_ap":           round(float(map_df["mean_average_precision"].median()), 4),
        "percent_replicating": round(float(map_df["below_corrected_p"].mean() * 100), 2),
        "n_compounds":         len(map_df),
    }


def compute_bio_recall(Z, meta, target_map, databases,
                       label_key="label_name", k=10):
    """
    Bio-recall: do compounds targeting the same biological pathway/complex/
    interacting proteins cluster together in embedding space?

    For each compound with known gene targets:
    1. Compute its median embedding
    2. Find its k nearest neighbour compounds (by median embedding cosine sim)
    3. Check if any neighbour shares a biological relationship via:
       - CORUM (same protein complex)
       - Reactome (same pathway)
       - hu.MAP (physical interaction)
       - StringDB (functional association)
    4. Recall@k = fraction of compounds with at least one related neighbour

    Parameters
    ----------
    Z : np.ndarray [N, D]
    meta : pd.DataFrame with label_key column
    target_map : dict {compound_name: [gene1, gene2, ...]}
    databases : dict with keys "corum", "reactome", "humap", "stringdb"
    label_key : str
    k : int — number of nearest neighbours

    Returns
    -------
    dict with recall@k for each database
    """
    from bio_databases import (genes_share_complex, genes_share_pathway,
                                genes_interact)

    labels = meta[label_key].astype(str).values

    # Compute median embedding per compound
    unique_compounds = [c for c in np.unique(labels) if c in target_map]
    if len(unique_compounds) < 2:
        return {f"bio_recall_{db}_at{k}": float("nan")
                for db in ["corum", "reactome", "humap", "stringdb"]}

    compound_embeddings = {}
    compound_targets = {}
    for c in unique_compounds:
        mask = labels == c
        compound_embeddings[c] = np.median(Z[mask], axis=0)
        compound_targets[c] = set(target_map[c])

    compounds = list(compound_embeddings.keys())
    Z_compounds = np.array([compound_embeddings[c] for c in compounds])

    # L2-normalize for cosine similarity
    norms = np.linalg.norm(Z_compounds, axis=1, keepdims=True)
    Z_norm = Z_compounds / np.maximum(norms, 1e-8)
    sim_matrix = Z_norm @ Z_norm.T

    # For each compound, find k nearest neighbours (excluding self)
    k_actual = min(k, len(compounds) - 1)
    recalls = {db: [] for db in ["corum", "reactome", "humap", "stringdb"]}

    for i, c in enumerate(compounds):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # exclude self
        neighbor_idx = np.argsort(sims)[-k_actual:][::-1]

        c_targets = compound_targets[c]

        for db_name, db_data in databases.items():
            found = False
            for j in neighbor_idx:
                n_targets = compound_targets[compounds[j]]
                # Check if any pair of (c_target, n_target) is related
                for g1 in c_targets:
                    for g2 in n_targets:
                        if g1 == g2:
                            continue
                        if db_name == "corum" and genes_share_complex(g1, g2, db_data):
                            found = True
                        elif db_name == "reactome" and genes_share_pathway(g1, g2, db_data):
                            found = True
                        elif db_name in ("humap", "stringdb") and genes_interact(g1, g2, db_data):
                            found = True
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            recalls[db_name].append(1.0 if found else 0.0)

    results = {}
    for db_name, hits in recalls.items():
        results[f"bio_recall_{db_name}_at{k}"] = round(float(np.mean(hits)), 4)
    results["bio_recall_n_compounds"] = len(compounds)

    return results


# --------------------------------------------------------------------------- #
# Replicate-tightness vs batch-spread ratio
# --------------------------------------------------------------------------- #
def compute_replicate_batch_ratio(Z, meta, label_key="label_name",
                                  batch_key="batch", control_key="DMSO",
                                  n_pairs=20000, seed=42):
    """
    Ratio of within-compound cosine distance to within-batch cross-compound
    cosine distance.

    For each treatment compound c with >=2 replicates:
        d_rep(c) = mean cosine_dist among replicates of c
    For each batch b:
        d_batch(b) = mean cosine_dist among random cross-compound pairs
                     drawn from non-control wells of b

    The reported scalar is mean(d_rep) / mean(d_batch).
    ratio < 1.0 means replicates are tighter than batch-mates (biology preserved).
    ratio = 1.0 means a method has flattened the distinction (overcorrection).
    ratio > 1.0 means batch-mates are tighter than replicates (broken).

    Controls are excluded from both the replicate set and the batch-spread set.
    """
    labels  = meta[label_key].astype(str).values
    batches = meta[batch_key].astype(str).values
    is_ctrl = labels == control_key
    Z_n     = _l2_normalize(Z)
    rng     = np.random.default_rng(seed)

    # Replicate distances per non-control compound
    rep_dists = []
    for c in np.unique(labels[~is_ctrl]):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        if len(idx) > 200:
            idx = rng.choice(idx, size=200, replace=False)
        sims = Z_n[idx] @ Z_n[idx].T
        triu = sims[np.triu_indices(len(idx), k=1)]
        rep_dists.extend((1.0 - triu).tolist())

    if not rep_dists:
        return {"replicate_batch_ratio": float("nan"),
                "mean_replicate_dist":   float("nan"),
                "mean_batch_dist":       float("nan")}

    # Within-batch cross-compound distances
    batch_dists = []
    for b in np.unique(batches):
        in_b = np.where((batches == b) & (~is_ctrl))[0]
        if len(in_b) < 2:
            continue
        n_sample = min(n_pairs // max(1, len(np.unique(batches))), len(in_b) * (len(in_b) - 1) // 2)
        a = rng.choice(in_b, size=n_sample, replace=True)
        c = rng.choice(in_b, size=n_sample, replace=True)
        keep = (a != c) & (labels[a] != labels[c])
        if keep.sum() == 0:
            continue
        sims = (Z_n[a[keep]] * Z_n[c[keep]]).sum(axis=1)
        batch_dists.extend((1.0 - sims).tolist())

    if not batch_dists:
        return {"replicate_batch_ratio": float("nan"),
                "mean_replicate_dist":   float(np.mean(rep_dists)),
                "mean_batch_dist":       float("nan")}

    mean_rep   = float(np.mean(rep_dists))
    mean_batch = float(np.mean(batch_dists))
    return {
        "replicate_batch_ratio": round(mean_rep / max(mean_batch, 1e-8), 4),
        "mean_replicate_dist":   round(mean_rep, 4),
        "mean_batch_dist":       round(mean_batch, 4),
    }


# --------------------------------------------------------------------------- #
# Edge-well bias removal
# --------------------------------------------------------------------------- #
def _classify_well(well_str):
    """Return 'corner' | 'edge' | 'center' for a 'A01'..'P24' well string."""
    if not isinstance(well_str, str) or len(well_str) < 2:
        return "unknown"
    row_letter = well_str[0].upper()
    try:
        col_num = int(well_str[1:])
    except ValueError:
        return "unknown"
    if row_letter not in "ABCDEFGHIJKLMNOP":
        return "unknown"
    row_idx = ord(row_letter) - ord("A") + 1
    row_edge = row_idx == 1 or row_idx == 16
    col_edge = col_num == 1 or col_num == 24
    if row_edge and col_edge:
        return "corner"
    if row_edge or col_edge:
        return "edge"
    return "center"


def compute_edge_well_bias(Z, meta, label_key="label_name",
                           well_key="Metadata_Well_str",
                           control_key="DMSO"):
    """
    Edge-well bias quantification.

    The 1536-well plate footprint has known edge effects: wells in row A/P or
    column 1/24 see different evaporation, illumination, and meniscus behaviour
    than center wells. A batch-correction method that ignores well position will
    leave a signature in the embedding: position class (edge/center) predicts
    compound identity even after correction.

    Metric: mean per-compound silhouette across well-position classes
            relative to the same silhouette on compound identity.
    A method that perfectly removes well-position bias produces
    position_silhouette ≈ 0 while keeping compound_silhouette intact.

    Returns also the per-class fraction of variance explained by well position
    via a 1-vs-rest centroid distance ratio.
    """
    if well_key not in meta.columns:
        return {"edge_well_silhouette":  float("nan"),
                "edge_well_centroid_d":  float("nan"),
                "edge_well_n_per_class": {}}

    pos_class = meta[well_key].apply(_classify_well).values
    labels    = meta[label_key].astype(str).values
    is_ctrl   = labels == control_key
    mask      = (~is_ctrl) & (pos_class != "unknown")

    if mask.sum() < 50 or len(np.unique(pos_class[mask])) < 2:
        return {"edge_well_silhouette":  float("nan"),
                "edge_well_centroid_d":  float("nan"),
                "edge_well_n_per_class": {}}

    sample_size = min(5000, mask.sum())
    sil = float(silhouette_score(Z[mask], pos_class[mask],
                                 sample_size=sample_size, random_state=42))

    Z_n = _l2_normalize(Z[mask])
    classes = np.unique(pos_class[mask])
    centroids = np.array([Z_n[pos_class[mask] == c].mean(axis=0) for c in classes])
    centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)
    inter_d = np.mean([
        1.0 - centroids[i] @ centroids[j]
        for i in range(len(classes)) for j in range(i + 1, len(classes))
    ])

    n_per_class = {c: int((pos_class[mask] == c).sum()) for c in classes}
    return {
        "edge_well_silhouette":  round(sil, 4),
        "edge_well_centroid_d":  round(float(inter_d), 4),
        "edge_well_n_per_class": n_per_class,
    }


# --------------------------------------------------------------------------- #
# Top-k retrieval saturation
# --------------------------------------------------------------------------- #
def compute_topk_recall_curve(Z, meta, label_key="label_name",
                              control_key="DMSO",
                              ks=(1, 5, 10, 25, 50, 100), seed=42):
    """
    Top-k retrieval saturation curve.

    For each query well q (non-control):
        rank its k nearest neighbours by cosine similarity (excluding q itself)
        hit@k(q) = 1 if any of those neighbours is the same compound as q

    Recall@k = mean(hit@k) over query wells.

    Returns {f'recall_at{k}': value, ...}. The shape of the curve over k
    distinguishes methods that rank ALL same-compound pairs tightly
    (steep rise to 1.0 at small k) from methods that recover only some
    replicates (slow saturation).
    """
    labels  = meta[label_key].astype(str).values
    is_ctrl = labels == control_key
    Z_n     = _l2_normalize(Z[~is_ctrl])
    y       = labels[~is_ctrl]
    N       = Z_n.shape[0]
    if N < 2:
        return {f"recall_at{k}": float("nan") for k in ks}

    k_max = min(max(ks), N - 1)
    nbrs = NearestNeighbors(n_neighbors=k_max + 1, metric="cosine", algorithm="brute")
    nbrs.fit(Z_n)
    _, idx = nbrs.kneighbors(Z_n)
    idx = idx[:, 1:]  # drop self

    results = {}
    for k in ks:
        k_eff = min(k, k_max)
        neigh_labels = y[idx[:, :k_eff]]
        hits = (neigh_labels == y[:, None]).any(axis=1)
        results[f"recall_at{k}"] = round(float(hits.mean()), 4)
    return results


def auroc_replicates(Z, meta, label_key="label_name",
                     min_reps=2, max_per_compound=300,
                     n_null=50_000, seed=42,
                     control_key=None, cross_batch=False, batch_key="batch"):
    """
    AUROC for replicate-vs-null pair classification using cosine similarity.

    Positive pairs : same compound (replicates)
    Negative pairs : random cross-compound pairs
    Score          : cosine similarity

    control_key : if given, rows with this label are dropped from BOTH pools before
                  pairing. DMSO is the negative control; counting DMSO-DMSO pairs as
                  "biology" inflates the score with control self-coherence, so the
                  default (None = include) over-credits methods that tighten the DMSO
                  cloud. Mirrors compute_map, which excludes controls.
    cross_batch : if True, positive pairs are restricted to same-compound wells from
                  DIFFERENT batches — the deployment-relevant signal (does a compound
                  look like itself ACROSS batches, not just within one). The default
                  (within+across) also rewards within-batch similarity, where there is
                  no batch effect to remove.

    Defaults reproduce the original behaviour exactly (control_key=None, cross_batch=False).
    Returns AUROC ∈ [0, 1]; 1.0 = perfect separation, 0.5 = random.
    """
    Z_norm  = _l2_normalize(Z)
    labels  = meta[label_key].astype(str).values
    batches = meta[batch_key].astype(str).values if cross_batch else None
    if control_key is not None:
        keep   = labels != control_key
        Z_norm = Z_norm[keep]
        labels = labels[keep]
        if batches is not None:
            batches = batches[keep]
    N      = Z_norm.shape[0]
    rng    = np.random.default_rng(seed)

    idx_a = rng.integers(0, N, size=n_null * 2)
    idx_b = rng.integers(0, N - 1, size=n_null * 2)
    idx_b[idx_b >= idx_a] += 1
    cross_mask = labels[idx_a] != labels[idx_b]
    null_sims  = (Z_norm[idx_a[cross_mask]] * Z_norm[idx_b[cross_mask]]).sum(axis=1)
    null_sims  = null_sims[:n_null]

    pos_sims = []
    for lbl in np.unique(labels):
        mask = np.where(labels == lbl)[0]
        if len(mask) < min_reps:
            continue
        if len(mask) > max_per_compound:
            mask = rng.choice(mask, size=max_per_compound, replace=False)
        Z_lbl = Z_norm[mask]
        sim   = Z_lbl @ Z_lbl.T
        iu    = np.triu_indices(len(Z_lbl), k=1)
        triu  = sim[iu]
        if cross_batch:
            b_lbl = batches[mask]
            triu  = triu[b_lbl[iu[0]] != b_lbl[iu[1]]]   # cross-batch positives only
        pos_sims.extend(triu.tolist())

    if not pos_sims:
        return {"auroc_replicating": float("nan")}

    pos_sims = np.array(pos_sims)
    scores   = np.concatenate([pos_sims, null_sims])
    y_true   = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(null_sims))])

    return {"auroc_replicating": round(float(roc_auc_score(y_true, scores)), 4)}


def auroc_replicates_anchor(Z_anchor, meta_anchor, Z_ref, meta_ref,
                            label_key="label_name", control_key="DMSO",
                            max_pairs_per_compound=30_000,
                            n_null=50_000, seed=42):
    """AUROC for anchor-vs-reference replicate retrieval (cosine similarity).

    Positive pairs : same compound, one well from the anchor set and one from the
                     reference set
    Negative pairs : random cross-compound (anchor, reference) pairs
    Score          : cosine similarity

    Companion to auroc_replicates, which pairs within a single pool. Here every
    pair spans the anchor/reference boundary by construction, so with anchor = a
    held-out batch and reference = the batches a correction was fitted on, every
    positive is a cross-batch pair: the deployment question of whether the new
    batch's compounds match themselves in the data you already have. The control
    is dropped from both sides by default, for the reasons the chapter ranks on
    the DMSO-excluded variant (control_key=None keeps it).
    Returns AUROC in [0, 1]; 0.5 = random.
    """
    Za, Zr = _l2_normalize(Z_anchor), _l2_normalize(Z_ref)
    la = meta_anchor[label_key].astype(str).values
    lr = meta_ref[label_key].astype(str).values
    if control_key is not None:
        ka, kr = la != control_key, lr != control_key
        Za, la = Za[ka], la[ka]
        Zr, lr = Zr[kr], lr[kr]
    if len(la) == 0 or len(lr) == 0:
        return {"auroc_replicating": float("nan")}
    rng = np.random.default_rng(seed)

    ia = rng.integers(0, len(la), size=n_null * 2)
    ir = rng.integers(0, len(lr), size=n_null * 2)
    cross = la[ia] != lr[ir]
    null_sims = (Za[ia[cross]] * Zr[ir[cross]]).sum(axis=1)[:n_null]

    pos_sims = []
    for lbl in np.unique(la):
        mr = np.where(lr == lbl)[0]
        if len(mr) == 0:
            continue
        sims = (Za[la == lbl] @ Zr[mr].T).ravel()
        if len(sims) > max_pairs_per_compound:
            sims = rng.choice(sims, size=max_pairs_per_compound, replace=False)
        pos_sims.extend(sims.tolist())
    if not pos_sims or len(null_sims) == 0:
        return {"auroc_replicating": float("nan")}

    scores = np.concatenate([np.asarray(pos_sims), null_sims])
    y_true = np.concatenate([np.ones(len(pos_sims)), np.zeros(len(null_sims))])
    return {"auroc_replicating": round(float(roc_auc_score(y_true, scores)), 4)}


# --------------------------------------------------------------------------- #
# Branch-level runner
# --------------------------------------------------------------------------- #
def probe_balanced_accuracy(Z_train, meta_train, Z_test, meta_test,
                            label_key="label_name", control_key=None,
                            seed=42, max_iter=2000, C=1.0):
    """Cross-set linear-probe balanced accuracy.

    Fit a multinomial logistic-regression probe on ``(Z_train, labels)``, predict
    ``Z_test``, and return balanced accuracy (mean per-class recall — robust to the
    DMSO class imbalance, unlike plain accuracy). Standardisation is fit on the
    train split only. If ``control_key`` is given, that label is dropped from both
    splits first (treatment-only accuracy). Returns NaN when the probe is undefined
    (fewer than two train classes, or an empty test split).

    In the deployment / LOBO setting ``Z_train`` is the corrected reference (the
    batches you already have) and ``Z_test`` the corrected held-out batch, so this
    measures whether the correction keeps compounds classifiable on an unseen batch.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score

    ytr = meta_train[label_key].astype(str).to_numpy()
    yte = meta_test[label_key].astype(str).to_numpy()
    Xtr, Xte = np.asarray(Z_train, dtype=float), np.asarray(Z_test, dtype=float)
    if control_key is not None:
        ktr, kte = ytr != str(control_key), yte != str(control_key)
        Xtr, ytr = Xtr[ktr], ytr[ktr]
        Xte, yte = Xte[kte], yte[kte]
    if len(np.unique(ytr)) < 2 or len(yte) == 0:
        return float("nan")

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=max_iter, C=C, random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))
    return float(balanced_accuracy_score(yte, pred))


def evaluate_branch(Z_full, meta_full, Z_sub, meta_sub,
                    batch_keys, lisi_perplexity, skip_sil_lisi=False,
                    leiden_k=15, leiden_resolution=1.0,
                    target_map=None, bio_databases=None, bio_recall_k=10):
    """
    Run all metrics for one embedding branch.

    Z_full / meta_full  — full dataset  (used for mAP + AUROC + bio-recall)
    Z_sub  / meta_sub   — subsampled    (used for silhouette / LISI / kBET / Leiden)
    skip_sil_lisi       — skip batch-mixing + clustering metrics (faster)
    target_map          — {compound: [gene1, ...]} from bio_annotations (optional)
    bio_databases       — dict from bio_databases.load_all_databases() (optional)

    Returns a dict with keys: silhouette, lisi, kbet, leiden, map, auroc, [bio_recall].
    """
    results = {}
    if not skip_sil_lisi:
        results["silhouette"] = compute_silhouette(Z_sub, meta_sub, batch_keys)
        results["lisi"]       = compute_lisi(Z_sub, meta_sub, batch_keys,
                                             perplexity=lisi_perplexity)
        kbet_results = {}
        for bk in batch_keys:
            kbet_results.update(compute_kbet(Z_sub, meta_sub, batch_key=bk))
        results["kbet"]   = kbet_results
        results["leiden"] = compute_leiden(Z_sub, meta_sub,
                                           k=leiden_k, resolution=leiden_resolution)
    results["map"]   = compute_map(Z_full, meta_full)
    results["auroc"] = auroc_replicates(Z_full, meta_full)
    # Corrected bio readouts: the headline auroc counts DMSO self-coherence and
    # within-batch pairs, which inflates control-tightening methods and penalises whiteners.
    # _nodmso excludes the control; _xbatch additionally restricts positives to DIFFERENT
    # batches (the deployment-relevant "looks like itself across batches"). DMSO is the
    # standard control (cf. compute_map); cross-batch needs the real 'batch' column.
    results["auroc_nodmso"] = auroc_replicates(Z_full, meta_full, control_key="DMSO")
    if "batch" in meta_full.columns:
        results["auroc_xbatch"] = auroc_replicates(
            Z_full, meta_full, control_key="DMSO", cross_batch=True, batch_key="batch")

    # Extras: replicate/batch ratio, edge-well bias, top-k saturation
    primary_bk = batch_keys[0] if batch_keys else "batch"
    results["replicate_batch"] = compute_replicate_batch_ratio(
        Z_full, meta_full, batch_key=primary_bk)
    results["edge_well"]       = compute_edge_well_bias(Z_full, meta_full)
    results["topk_recall"]     = compute_topk_recall_curve(Z_full, meta_full)

    if target_map is not None and bio_databases is not None:
        results["bio_recall"] = compute_bio_recall(
            Z_full, meta_full, target_map, bio_databases, k=bio_recall_k)

    return results
