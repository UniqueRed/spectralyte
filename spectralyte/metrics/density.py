"""
spectralyte/metrics/density.py
================================
Density distribution metric for embedding spaces.

Measures how uniformly embeddings are distributed across the vector space.
A well-distributed space has embeddings spread evenly — every region is
similarly populated. A poorly distributed space has tight dense clusters
separated by large empty voids. Queries near cluster boundaries flip
between neighborhoods with small changes in phrasing, causing inconsistent
retrieval results.

Two complementary measures are computed:

1. k-NN Distance Distribution:
   For each embedding, compute the distance to its k-th nearest neighbor.
   The coefficient of variation (CV = std / mean) of these distances
   characterizes the density structure.

   - Low CV  → uniform distribution (healthy)
   - High CV → clustering with voids (problematic)

2. Local Outlier Factor (LOF):
   Compares each point's local density to its neighbors' local densities.
   LOF ≈ 1.0 → normal point (consistent local density)
   LOF >> 1.0 → outlier in sparse region
   LOF << 1.0 → point in unusually dense region

High-dimensional note:
   For d > 100, Euclidean distance degrades (curse of dimensionality).
   The implementation L2-normalizes embeddings and uses Euclidean distance
   on the normalized vectors, which is mathematically equivalent to
   cosine distance: ||u - v||^2 = 2(1 - cos(u,v)) for unit vectors.

Reference:
    Breunig, M. et al. (2000). LOF: Identifying Density-Based Local Outliers.
    SIGMOD Record, 29(2), 93-104.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy.spatial import cKDTree
from sklearn.neighbors import LocalOutlierFactor


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class DensityResult:
    """
    Result of a density distribution computation.

    Attributes
    ----------
    cv : float
        Coefficient of variation of k-NN distances (std / mean).
        Low = uniform distribution. High = clustering.
    mean_knn_distance : float
        Mean distance to k-th nearest neighbor across all embeddings.
    std_knn_distance : float
        Standard deviation of k-NN distances.
    knn_distances : np.ndarray
        Per-embedding distance to k-th nearest neighbor. Shape (n,).
    lof_scores : np.ndarray
        Local Outlier Factor score per embedding. Shape (n,).
        Near 1.0 = normal. >> 1.0 = sparse outlier.
    n_outliers : int
        Number of embeddings with LOF score above lof_threshold.
    outlier_indices : np.ndarray
        Indices of outlier embeddings. Shape (n_outliers,).
    interpretation : str
        Human-readable severity: 'uniform', 'moderate', 'clustered', 'severe'.
    k : int
        Number of nearest neighbors used.
    lof_threshold : float
        LOF threshold used to define outliers.
    n_vectors : int
        Number of embedding vectors analyzed.
    n_dims : int
        Embedding dimensionality.
    normalized : bool
        True if embeddings were L2-normalized before distance computation.
    """

    cv: float
    mean_knn_distance: float
    std_knn_distance: float
    knn_distances: np.ndarray = field(repr=False)
    lof_scores: np.ndarray = field(repr=False)
    n_outliers: int
    outlier_indices: np.ndarray = field(repr=False)
    interpretation: str
    k: int
    lof_threshold: float
    n_vectors: int
    n_dims: int
    normalized: bool


# ── Interpretation ─────────────────────────────────────────────────────────────

def _interpret(cv: float) -> str:
    """
    Map CV of k-NN distances to a human-readable density assessment.

    Parameters
    ----------
    cv : float
        Coefficient of variation. Range [0, inf).

    Returns
    -------
    str
        One of 'uniform', 'moderate', 'clustered', 'severe'.
    """
    if cv < 0.3:
        return "uniform"
    elif cv < 0.6:
        return "moderate"
    elif cv < 1.0:
        return "clustered"
    else:
        return "severe"


# ── Normalization ──────────────────────────────────────────────────────────────

def _should_normalize(d: int) -> bool:
    """
    Determine whether to L2-normalize based on embedding dimensionality.

    For d > 100, Euclidean distance in the original space is unreliable
    due to the curse of dimensionality. L2-normalizing and using Euclidean
    distance on unit vectors is equivalent to cosine distance, which is
    more meaningful in high dimensions.

    Parameters
    ----------
    d : int
        Embedding dimensionality.

    Returns
    -------
    bool
        True if L2-normalization should be applied before distance computation.
    """
    return d > 100


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows. Zero vectors are left as-is (avoid div by zero)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


# ── k-NN distance computation ──────────────────────────────────────────────────

def _compute_knn_distances(
    embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Compute the distance to the k-th nearest neighbor for each embedding.

    Uses scipy.spatial.cKDTree for efficient computation. The k+1 query
    accounts for each point being its own nearest neighbor (distance 0),
    which is excluded from the result.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (n, d). Should be L2-normalized if d > 100.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    np.ndarray
        Distance to k-th nearest neighbor for each point. Shape (n,).
    """
    tree = cKDTree(embeddings)
    # Query k+1 neighbors: index 0 is the point itself (distance 0)
    # We want the k-th actual neighbor, which is at index k in the result
    distances, _ = tree.query(embeddings, k=k + 1)
    # distances shape: (n, k+1). Column 0 is self (dist=0), column k is k-th neighbor
    return distances[:, k]


# ── LOF computation ────────────────────────────────────────────────────────────

def _compute_lof_scores(
    embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Compute Local Outlier Factor scores for each embedding.

    LOF measures the local density deviation of each point relative to
    its neighbors. A score near 1.0 indicates a normal point. A score
    significantly above 1.0 indicates a point in a sparse region (outlier).

    sklearn's LOF returns negative scores (higher anomaly = more negative).
    We negate and shift so that normal points have score ~1.0 and outliers
    have scores > 1.0, matching the standard LOF definition.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (n, d).
    k : int
        Number of neighbors for LOF computation (n_neighbors parameter).

    Returns
    -------
    np.ndarray
        LOF score per embedding. Shape (n,). Values >= 0.
        Near 1.0 = normal density. >> 1.0 = sparse outlier.
    """
    lof = LocalOutlierFactor(n_neighbors=k, metric='euclidean')
    lof.fit(embeddings)
    # sklearn returns negative_outlier_factor_ — more negative = more anomalous
    # Standard LOF score = -1 * negative_outlier_factor_
    lof_scores = -lof.negative_outlier_factor_
    return lof_scores


# ── Core computation ───────────────────────────────────────────────────────────

def compute(
    embeddings: np.ndarray,
    k: int = 10,
    lof_threshold: float = 1.5,
    sample_size: Optional[int] = 10_000,
    random_seed: int = 42,
) -> DensityResult:
    """
    Compute the density distribution of an embedding space.

    Measures how uniformly embeddings are distributed using two metrics:
    (1) the coefficient of variation of k-nearest-neighbor distances, and
    (2) Local Outlier Factor scores.

    A uniform space has all embeddings at similar distances from their
    neighbors (low CV). A clustered space has some embeddings tightly
    packed (small k-NN distances) and others isolated (large k-NN
    distances), producing high CV. The LOF scores identify which specific
    embeddings are in anomalously sparse or dense regions.

    For d > 100 dimensions, embeddings are L2-normalized before distance
    computation. This is equivalent to using cosine distance, which is more
    discriminative than Euclidean distance in high dimensions.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (n, d). Any embedding model or dimension.
    k : int
        Number of nearest neighbors. Default 10.
        Should be smaller than the smallest expected cluster size.
    lof_threshold : float
        LOF score above which a point is classified as an outlier.
        Default 1.5. Increase to be more conservative about outlier labeling.
    sample_size : Optional[int]
        If set, subsample for LOF computation (expensive for large n).
        k-NN distances are always computed on the full index.
        Default 10,000. Set to None to use all vectors.
    random_seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    DensityResult
        Dataclass with CV, k-NN distances, LOF scores, outlier indices,
        interpretation, and metadata.

    Raises
    ------
    ValueError
        If embeddings is not 2D, or k >= n.

    Example
    -------
    >>> import numpy as np
    >>> from spectralyte.metrics.density import compute
    >>> embeddings = np.random.randn(1000, 384)
    >>> result = compute(embeddings)
    >>> print(f"CV: {result.cv:.3f} — {result.interpretation}")
    >>> print(f"Outliers: {result.n_outliers}")
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D array of shape (n, d), got shape {embeddings.shape}"
        )

    n, d = embeddings.shape

    if k >= n:
        raise ValueError(
            f"k ({k}) must be less than number of embeddings ({n})"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # ── Normalization decision ─────────────────────────────────────────────────
    normalized = _should_normalize(d)
    working = _l2_normalize(embeddings) if normalized else embeddings.copy()

    # ── k-NN distances on full index ───────────────────────────────────────────
    knn_distances = _compute_knn_distances(working, k)

    # ── Coefficient of variation ───────────────────────────────────────────────
    mean_dist = float(np.mean(knn_distances))
    std_dist = float(np.std(knn_distances))

    # Guard against degenerate case where all distances are zero
    if mean_dist == 0:
        cv = 0.0
    else:
        cv = std_dist / mean_dist

    # ── LOF on sample ──────────────────────────────────────────────────────────
    # LOF is O(n^2) in the worst case — sample for large indices
    if sample_size is not None and n > sample_size:
        rng = np.random.RandomState(random_seed)
        sample_indices = rng.choice(n, size=sample_size, replace=False)
        lof_input = working[sample_indices]
        lof_scores_sample = _compute_lof_scores(lof_input, k=min(k, sample_size - 1))
        # Expand back to full index — unsampled points get score 1.0 (neutral)
        lof_scores = np.ones(n)
        lof_scores[sample_indices] = lof_scores_sample
    else:
        lof_scores = _compute_lof_scores(working, k=k)

    # ── Outlier identification ─────────────────────────────────────────────────
    outlier_mask = lof_scores > lof_threshold
    outlier_indices = np.where(outlier_mask)[0]
    n_outliers = int(outlier_mask.sum())

    return DensityResult(
        cv=float(cv),
        mean_knn_distance=mean_dist,
        std_knn_distance=std_dist,
        knn_distances=knn_distances,
        lof_scores=lof_scores,
        n_outliers=n_outliers,
        outlier_indices=outlier_indices,
        interpretation=_interpret(cv),
        k=k,
        lof_threshold=lof_threshold,
        n_vectors=n,
        n_dims=d,
        normalized=normalized,
    )