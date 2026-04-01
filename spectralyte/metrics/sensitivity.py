"""
spectralyte/metrics/sensitivity.py
=====================================
Retrieval Sensitivity Index (RSI) for embedding spaces.

Measures how stable the retrieval results are under small perturbations
of query vectors. A stable region returns the same documents regardless
of minor query rephrasing. A brittle region returns completely different
documents when the query changes slightly.

The RSI quantifies this at the local level — producing a stability score
for each embedding in the index, identifying which specific documents
live in brittle zones where retrieval is unreliable.

Algorithm:
    For each embedding v_i (or a representative sample):
        1. Generate m perturbed copies:
              v_i^(j) = L2_normalize(v_i + epsilon * noise_j)
              where noise_j ~ N(0, I_d)
              and epsilon = epsilon_fraction * mean_knn_distance
        2. Compute top-k nearest neighbors for v_i and each v_i^(j)
        3. Stability score for v_i:
              stability_i = (1/m) * sum_j Jaccard(kNN(v_i), kNN(v_i^(j)))
              Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    Global RSI = mean(stability_i) across all sampled embeddings.

Key design decisions:
    - Epsilon is scaled to mean k-NN distance, not absolute, so perturbations
      are meaningful across different regions and embedding models.
    - Perturbed vectors are L2-normalized to remain on the unit hypersphere.
    - Sampling is used for large indices to keep runtime tractable.
    - Unsampled embeddings are assigned the mean stability of their nearest
      sampled neighbors (local interpolation).

Reference:
    Jaccard similarity: Jaccard, P. (1912). The distribution of the flora
    in the alpine zone. New Phytologist, 11(2), 37-50.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Set
from scipy.spatial import cKDTree


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class SensitivityResult:
    """
    Result of a Retrieval Sensitivity Index computation.

    Attributes
    ----------
    mean_stability : float
        Mean Jaccard stability across all embeddings. Range [0, 1].
        1.0 = perfectly stable (perturbation never changes results).
        0.0 = completely unstable (results change entirely with perturbation).
    stability_per_embedding : np.ndarray
        Per-embedding stability score. Shape (n,).
        Embeddings with low scores are in brittle zones.
    brittle_zone_indices : np.ndarray
        Indices of embeddings with stability below brittle_threshold.
    n_brittle : int
        Number of embeddings in brittle zones.
    brittle_fraction : float
        Fraction of index in brittle zones. Range [0, 1].
    epsilon_used : float
        Absolute epsilon value used for perturbations (in embedding units).
    epsilon_fraction : float
        Epsilon as fraction of mean k-NN distance.
    mean_knn_distance : float
        Mean k-NN distance used to calibrate epsilon.
    interpretation : str
        Human-readable stability assessment.
    k : int
        Number of nearest neighbors used.
    m : int
        Number of perturbations per embedding.
    brittle_threshold : float
        Stability threshold below which an embedding is 'brittle'.
    n_vectors : int
        Total number of embedding vectors in the index.
    n_sampled : int
        Number of embeddings for which RSI was directly computed.
    n_dims : int
        Embedding dimensionality.
    sampled : bool
        True if RSI was computed on a sample rather than the full index.
    """

    mean_stability: float
    stability_per_embedding: np.ndarray = field(repr=False)
    brittle_zone_indices: np.ndarray = field(repr=False)
    n_brittle: int
    brittle_fraction: float
    epsilon_used: float
    epsilon_fraction: float
    mean_knn_distance: float
    interpretation: str
    k: int
    m: int
    brittle_threshold: float
    n_vectors: int
    n_sampled: int
    n_dims: int
    sampled: bool


# ── Interpretation ─────────────────────────────────────────────────────────────

def _interpret(mean_stability: float) -> str:
    """
    Map mean stability to a human-readable assessment.

    Parameters
    ----------
    mean_stability : float
        Mean Jaccard stability. Range [0, 1].

    Returns
    -------
    str
        One of 'stable', 'moderate', 'sensitive', 'brittle'.
    """
    if mean_stability >= 0.80:
        return "stable"
    elif mean_stability >= 0.60:
        return "moderate"
    elif mean_stability >= 0.40:
        return "sensitive"
    else:
        return "brittle"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or matrix. Zero vectors are left unchanged."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return v / norms


def _jaccard(set_a: Set[int], set_b: Set[int]) -> float:
    """
    Compute Jaccard similarity between two sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns 1.0 if both sets are empty (convention for degenerate case).

    Parameters
    ----------
    set_a, set_b : Set[int]
        Sets of neighbor indices to compare.

    Returns
    -------
    float
        Jaccard similarity in [0, 1].
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _compute_mean_knn_distance(
    embeddings: np.ndarray,
    k: int,
) -> float:
    """
    Compute mean distance to k-th nearest neighbor across all embeddings.

    Used to calibrate epsilon so perturbations are proportional to the
    local geometry rather than being an absolute fixed value.

    Parameters
    ----------
    embeddings : np.ndarray
        L2-normalized embedding matrix of shape (n, d).
    k : int
        Number of nearest neighbors.

    Returns
    -------
    float
        Mean k-NN distance across all embeddings.
    """
    tree = cKDTree(embeddings)
    # k+1 because each point is its own nearest neighbor (distance 0)
    distances, _ = tree.query(embeddings, k=k + 1)
    # distances[:, k] is the distance to the k-th actual neighbor
    return float(np.mean(distances[:, k]))


def _stability_for_embedding(
    idx: int,
    embeddings: np.ndarray,
    tree: cKDTree,
    k: int,
    m: int,
    epsilon: float,
    rng: np.random.RandomState,
) -> float:
    """
    Compute stability score for a single embedding.

    Parameters
    ----------
    idx : int
        Index of the embedding to evaluate.
    embeddings : np.ndarray
        Full L2-normalized embedding matrix. Shape (n, d).
    tree : cKDTree
        Pre-built k-d tree over embeddings for efficient kNN queries.
    k : int
        Number of nearest neighbors.
    m : int
        Number of perturbations to generate.
    epsilon : float
        Absolute noise magnitude (pre-calibrated to mean k-NN distance).
    rng : np.random.RandomState
        Random number generator for reproducibility.

    Returns
    -------
    float
        Stability score in [0, 1]. Higher = more stable.
    """
    v = embeddings[idx]
    d = v.shape[0]

    # Get original k nearest neighbors (exclude self at index 0)
    _, orig_indices = tree.query(v, k=k + 1)
    orig_neighbors = set(orig_indices[1:])   # exclude self

    jaccard_scores = []

    for _ in range(m):
        # Generate perturbation: Gaussian noise scaled by epsilon
        noise = rng.randn(d)
        noise = noise / np.linalg.norm(noise) if np.linalg.norm(noise) > 0 else noise
        perturbed = v + epsilon * noise

        # L2-normalize to stay on the unit hypersphere
        perturbed = _l2_normalize(perturbed)

        # Get k nearest neighbors of perturbed vector
        _, pert_indices = tree.query(perturbed, k=k + 1)
        pert_neighbors = set(pert_indices[1:])   # exclude nearest (self or similar)

        jaccard_scores.append(_jaccard(orig_neighbors, pert_neighbors))

    return float(np.mean(jaccard_scores))


# ── Core computation ───────────────────────────────────────────────────────────

def compute(
    embeddings: np.ndarray,
    k: int = 10,
    m: int = 5,
    epsilon_fraction: float = 0.05,
    brittle_threshold: float = 0.5,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> SensitivityResult:
    """
    Compute the Retrieval Sensitivity Index (RSI) for an embedding space.

    For each embedding (or a representative sample), generates m perturbed
    copies by adding small calibrated Gaussian noise. Computes the Jaccard
    similarity between the original top-k result set and each perturbed
    result set. The mean Jaccard score across all perturbations is the
    stability score for that embedding.

    Epsilon is calibrated as a fraction of the mean k-NN distance across
    the index. This ensures perturbations are proportional to the local
    geometry — neither too small to matter nor too large to be unrealistic.

    For unsampled embeddings, stability is estimated by averaging the scores
    of their nearest sampled neighbors (local interpolation). This provides
    reasonable coverage of the full index without requiring full computation.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (n, d). Should be pre-normalized or will
        be L2-normalized internally.
    k : int
        Number of nearest neighbors for retrieval. Default 10.
    m : int
        Number of perturbations per embedding. More = more accurate but slower.
        Default 5.
    epsilon_fraction : float
        Noise magnitude as fraction of mean k-NN distance. Default 0.05.
        Represents a ~5% perturbation of the typical neighbor spacing.
    brittle_threshold : float
        Stability score below which an embedding is classified as brittle.
        Default 0.5.
    sample_size : Optional[int]
        Number of embeddings for which to compute RSI directly. If None,
        uses 20% of the index (min 100, max 5000). Set to -1 to compute
        for all embeddings.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    SensitivityResult
        Dataclass with mean stability, per-embedding scores, brittle zone
        indices, epsilon calibration info, and metadata.

    Raises
    ------
    ValueError
        If embeddings is not 2D, k >= n, m < 1, or epsilon_fraction <= 0.

    Example
    -------
    >>> import numpy as np
    >>> from spectralyte.metrics.sensitivity import compute
    >>> embeddings = np.random.randn(500, 384)
    >>> result = compute(embeddings)
    >>> print(f"Mean stability: {result.mean_stability:.3f}")
    >>> print(f"Brittle zones: {result.n_brittle} embeddings")
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D array of shape (n, d), got shape {embeddings.shape}"
        )

    n, d = embeddings.shape

    if k >= n:
        raise ValueError(f"k ({k}) must be less than n ({n})")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if epsilon_fraction <= 0:
        raise ValueError(f"epsilon_fraction must be > 0, got {epsilon_fraction}")
    if brittle_threshold < 0 or brittle_threshold > 1:
        raise ValueError(f"brittle_threshold must be in [0, 1], got {brittle_threshold}")

    rng = np.random.RandomState(random_seed)

    # ── L2 normalize ──────────────────────────────────────────────────────────
    V = _l2_normalize(embeddings)

    # ── Build k-d tree ────────────────────────────────────────────────────────
    tree = cKDTree(V)

    # ── Calibrate epsilon ──────────────────────────────────────────────────────
    mean_knn_dist = _compute_mean_knn_distance(V, k)
    epsilon = epsilon_fraction * mean_knn_dist

    # Guard against degenerate case (all identical embeddings)
    if epsilon == 0:
        epsilon = 1e-6

    # ── Determine sample ──────────────────────────────────────────────────────
    if sample_size == -1:
        # Compute for all embeddings
        sample_indices = np.arange(n)
        sampled = False
    else:
        if sample_size is None:
            # Default: 20% of index, clamped to [100, 5000]
            default_size = max(100, min(5000, int(n * 0.20)))
            sample_size = default_size

        if sample_size >= n:
            sample_indices = np.arange(n)
            sampled = False
        else:
            sample_indices = rng.choice(n, size=sample_size, replace=False)
            sampled = True

    n_sampled = len(sample_indices)

    # ── Compute RSI for sampled embeddings ────────────────────────────────────
    sample_stability = np.zeros(n_sampled)

    for i, idx in enumerate(sample_indices):
        sample_stability[i] = _stability_for_embedding(
            idx, V, tree, k, m, epsilon, rng
        )

    # ── Interpolate for unsampled embeddings ──────────────────────────────────
    stability_per_embedding = np.zeros(n)
    stability_per_embedding[sample_indices] = sample_stability

    if sampled:
        # For unsampled embeddings: mean stability of k nearest sampled neighbors
        # Build a tree of just the sampled embeddings for interpolation
        sampled_embeddings = V[sample_indices]
        sampled_tree = cKDTree(sampled_embeddings)

        unsampled_mask = np.ones(n, dtype=bool)
        unsampled_mask[sample_indices] = False
        unsampled_indices = np.where(unsampled_mask)[0]

        if len(unsampled_indices) > 0:
            # Find nearest sampled neighbors for each unsampled embedding
            interp_k = min(3, n_sampled)
            _, nn_in_sample = sampled_tree.query(V[unsampled_indices], k=interp_k)
            # Average their stability scores
            if interp_k == 1:
                stability_per_embedding[unsampled_indices] = sample_stability[nn_in_sample]
            else:
                stability_per_embedding[unsampled_indices] = sample_stability[nn_in_sample].mean(axis=1)

    # ── Global RSI ─────────────────────────────────────────────────────────────
    mean_stability = float(np.mean(stability_per_embedding))

    # ── Brittle zone identification ────────────────────────────────────────────
    brittle_mask = stability_per_embedding < brittle_threshold
    brittle_zone_indices = np.where(brittle_mask)[0]
    n_brittle = int(brittle_mask.sum())
    brittle_fraction = n_brittle / n

    return SensitivityResult(
        mean_stability=mean_stability,
        stability_per_embedding=stability_per_embedding,
        brittle_zone_indices=brittle_zone_indices,
        n_brittle=n_brittle,
        brittle_fraction=brittle_fraction,
        epsilon_used=epsilon,
        epsilon_fraction=epsilon_fraction,
        mean_knn_distance=mean_knn_dist,
        interpretation=_interpret(mean_stability),
        k=k,
        m=m,
        brittle_threshold=brittle_threshold,
        n_vectors=n,
        n_sampled=n_sampled,
        n_dims=d,
        sampled=sampled,
    )