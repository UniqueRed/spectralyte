"""
spectralyte/metrics/dimensionality.py
=======================================
Effective dimensionality metric for embedding spaces.

Embedding models produce high-dimensional vectors — 384, 768, 1536 dimensions
are common. But the actual number of dimensions carrying meaningful information
is often far lower. If your 1536-dim embedding space effectively uses only 40
dimensions, distinct documents get compressed together in the remaining space,
causing false positive retrievals.

This module computes two complementary measures of effective dimensionality:

1. Threshold-based effective dimensionality (d_eff):
   The minimum number of principal components needed to explain a given
   fraction of total variance (default 95%). Derived from SVD.

   Mathematical formulation:
       V_centered = V - mean(V, axis=0)
       V_centered = U @ S @ W.T         # SVD decomposition
       explained_variance_i = s_i^2 / sum(s_j^2)
       d_eff = min{k : sum_{i=1}^{k} explained_variance_i >= threshold}

2. Participation ratio (PR):
   A continuous measure of the effective number of dimensions. Ranges from
   1 (all variance in one dimension) to d (perfectly uniform variance).

   Mathematical formulation:
       lambda_i = s_i^2 / n             # eigenvalues of covariance matrix
       PR = (sum_i lambda_i)^2 / sum_i(lambda_i^2)

Reference:
    Explained variance / PCA: standard linear algebra.
    Participation ratio: Roy & Bhalla (2007), PLoS Computational Biology.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.decomposition import TruncatedSVD


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class DimensionalityResult:
    """
    Result of an effective dimensionality computation.

    Attributes
    ----------
    effective_dims : int
        Minimum principal components explaining variance_threshold of variance.
    nominal_dims : int
        Nominal embedding dimensionality (d in shape (n, d)).
    utilization : float
        effective_dims / nominal_dims. Low values indicate wasted dimensions.
    participation_ratio : float
        Continuous effective dimensionality measure. Range: [1, nominal_dims].
    explained_variance_ratio : np.ndarray
        Per-component explained variance ratios, sorted descending.
    cumulative_variance : np.ndarray
        Cumulative explained variance — useful for scree plot.
    variance_threshold : float
        The threshold used to determine effective_dims.
    interpretation : str
        Human-readable assessment of dimensionality health.
    n_vectors : int
        Number of embedding vectors analyzed.
    n_dims : int
        Embedding dimensionality (same as nominal_dims).
    truncated : bool
        True if TruncatedSVD was used (large input) rather than full SVD.
    """

    effective_dims: int
    nominal_dims: int
    utilization: float
    participation_ratio: float
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    variance_threshold: float
    interpretation: str
    n_vectors: int
    n_dims: int
    truncated: bool


# ── Interpretation ─────────────────────────────────────────────────────────────

def _interpret(utilization: float) -> str:
    """
    Map utilization ratio to a human-readable assessment.

    Parameters
    ----------
    utilization : float
        effective_dims / nominal_dims. Range [0, 1].

    Returns
    -------
    str
        One of 'healthy', 'moderate', 'low', 'critical'.
    """
    if utilization >= 0.20:
        return "healthy"
    elif utilization >= 0.10:
        return "moderate"
    elif utilization >= 0.04:
        return "low"
    else:
        return "critical"


# ── Core computation ───────────────────────────────────────────────────────────

def compute(
    embeddings: np.ndarray,
    variance_threshold: float = 0.95,
    truncation_threshold: int = 10_000,
    random_seed: int = 42,
) -> DimensionalityResult:
    """
    Compute the effective dimensionality of an embedding space.

    Effective dimensionality is the minimum number of principal components
    needed to explain a given fraction of the total variance in the embedding
    space. A 1536-dimensional embedding with effective dimensionality of 40
    means 97.4% of dimensions carry negligible information — distinct content
    collapses together in the compressed effective space, causing false positive
    retrievals.

    Two measures are computed:

    1. Threshold-based effective dimensionality (d_eff):
       Uses SVD of the centered embedding matrix. The explained variance ratio
       for component i is s_i^2 / sum(s_j^2). d_eff is the minimum k such
       that the cumulative sum of the top-k ratios >= variance_threshold.

    2. Participation ratio (PR):
       PR = (sum_i lambda_i)^2 / sum_i(lambda_i^2)
       where lambda_i = s_i^2 / n are the eigenvalues of the covariance matrix.
       PR is a continuous analog of effective dimensionality — less sensitive
       to the choice of threshold.

    For large inputs (n > truncation_threshold), TruncatedSVD is used to
    compute only the top min(n, d) singular values efficiently.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (n, d). Does not need to be pre-normalized.
    variance_threshold : float
        Fraction of variance that must be explained. Default 0.95.
        Range (0, 1].
    truncation_threshold : int
        Use TruncatedSVD for inputs with n > this value. Default 10,000.
    random_seed : int
        Random seed for TruncatedSVD reproducibility.

    Returns
    -------
    DimensionalityResult
        Dataclass with effective_dims, participation_ratio, explained variance
        curve, interpretation, and metadata.

    Raises
    ------
    ValueError
        If embeddings is not 2D, or variance_threshold is not in (0, 1].

    Example
    -------
    >>> import numpy as np
    >>> from spectralyte.metrics.dimensionality import compute
    >>> embeddings = np.random.randn(1000, 1536)
    >>> result = compute(embeddings)
    >>> print(f"{result.effective_dims} / {result.nominal_dims} dims used")
    >>> print(f"Utilization: {result.utilization:.1%}")
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D array of shape (n, d), got shape {embeddings.shape}"
        )
    if not (0 < variance_threshold <= 1.0):
        raise ValueError(
            f"variance_threshold must be in (0, 1], got {variance_threshold}"
        )

    n, d = embeddings.shape

    if n < 2:
        raise ValueError(f"Need at least 2 embeddings, got {n}")

    # ── Center the embeddings ──────────────────────────────────────────────────
    # SVD on centered data is equivalent to PCA.
    V = embeddings - embeddings.mean(axis=0)

    # ── SVD ───────────────────────────────────────────────────────────────────
    # For large n, TruncatedSVD is dramatically faster than full SVD.
    # We compute at most min(n, d) components — the maximum meaningful rank.
    max_components = min(n - 1, d)
    truncated = False

    if n > truncation_threshold:
        truncated = True
        n_components = min(max_components, 500)   # cap for speed
        svd = TruncatedSVD(n_components=n_components, random_state=random_seed)
        svd.fit(V)
        singular_values = svd.singular_values_
        explained_variance_ratio = svd.explained_variance_ratio_
    else:
        # Full SVD — exact solution
        # Use numpy's SVD with compute_uv=False since we only need singular values
        singular_values = np.linalg.svd(V, compute_uv=False)
        # Clip to non-negative (numerical noise can produce tiny negatives)
        singular_values = np.clip(singular_values, 0, None)
        # Explained variance from singular values
        variance = singular_values ** 2
        total_variance = variance.sum()
        if total_variance == 0:
            # Degenerate case: all embeddings are identical
            explained_variance_ratio = np.zeros(len(singular_values))
            explained_variance_ratio[0] = 1.0
        else:
            explained_variance_ratio = variance / total_variance

    # ── Cumulative variance ────────────────────────────────────────────────────
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # ── Effective dimensionality ───────────────────────────────────────────────
    # Find minimum k such that cumulative variance >= threshold
    above_threshold = np.where(cumulative_variance >= variance_threshold)[0]
    if len(above_threshold) == 0:
        # Threshold not reached within computed components (TruncatedSVD case)
        effective_dims = len(explained_variance_ratio)
    else:
        effective_dims = int(above_threshold[0]) + 1   # +1 for 1-indexed count

    # ── Participation ratio ────────────────────────────────────────────────────
    # lambda_i = s_i^2 / n  (eigenvalues of the covariance matrix)
    # PR = (sum lambda_i)^2 / sum(lambda_i^2)
    lambdas = (singular_values ** 2) / n
    lambda_sum = lambdas.sum()
    lambda_sum_sq = (lambdas ** 2).sum()

    if lambda_sum_sq == 0:
        participation_ratio = 1.0
    else:
        participation_ratio = float((lambda_sum ** 2) / lambda_sum_sq)

    # ── Utilization ───────────────────────────────────────────────────────────
    utilization = effective_dims / d

    return DimensionalityResult(
        effective_dims=effective_dims,
        nominal_dims=d,
        utilization=utilization,
        participation_ratio=participation_ratio,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_variance=cumulative_variance,
        variance_threshold=variance_threshold,
        interpretation=_interpret(utilization),
        n_vectors=n,
        n_dims=d,
        truncated=truncated,
    )