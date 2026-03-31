"""
spectralyte/metrics/anisotropy.py
==================================
Anisotropy metric for embedding spaces.

Anisotropy measures how uniformly vectors are distributed across the unit
hypersphere. A perfectly isotropic space has vectors pointing equally in all
directions — cosine similarity is maximally discriminative. A highly anisotropic
space has vectors clustered along a small number of directions — cosine similarity
loses its ability to distinguish between documents.

Mathematical formulation (Gram matrix):
    Given L2-normalized embedding matrix V of shape (n, d):
    G = V @ V.T                          # Gram matrix, shape (n, n)
    score = (G.sum() - G.trace()) / (n * (n - 1))

    This equals the mean pairwise cosine similarity across all (n choose 2) pairs.
    Score near 0 → isotropic (healthy).
    Score near 1 → fully anisotropic (severe).

Reference:
    Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations?
    arXiv:1909.00512
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class AnisotropyResult:
    """
    Result of an anisotropy computation.

    Attributes
    ----------
    score : float
        Mean pairwise cosine similarity. Bounded [0, 1].
        0 = perfectly isotropic. 1 = fully anisotropic.
    eigenvalues : np.ndarray
        Eigenvalues of the Gram matrix, sorted descending.
        Concentration in a few eigenvalues indicates anisotropy.
    interpretation : str
        Human-readable severity: 'healthy', 'moderate', 'high', or 'severe'.
    n_vectors : int
        Number of embedding vectors analyzed.
    n_dims : int
        Embedding dimensionality.
    sampled : bool
        True if the score was computed on a sample rather than the full index.
    sample_size : int
        Number of vectors used for Gram matrix computation.
    """

    score: float
    eigenvalues: np.ndarray
    interpretation: str
    n_vectors: int
    n_dims: int
    sampled: bool
    sample_size: int


# ── Interpretation thresholds ──────────────────────────────────────────────────

_THRESHOLDS = [
    (0.2, "healthy"),
    (0.4, "moderate"),
    (0.6, "high"),
    (1.1, "severe"),   # upper sentinel — catches everything >= 0.6
]


def _interpret(score: float) -> str:
    """Map anisotropy score to a human-readable severity label."""
    for threshold, label in _THRESHOLDS:
        if score < threshold:
            return label
    return "severe"


# ── Core computation ───────────────────────────────────────────────────────────

def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows of an embedding matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def compute(
    embeddings: np.ndarray,
    sample_size: Optional[int] = 5000,
    random_seed: int = 42,
) -> AnisotropyResult:
    """
    Compute the anisotropy of an embedding space.

    Anisotropy measures how uniformly vectors are distributed across the unit
    hypersphere. High anisotropy indicates that cosine similarity has lost
    discriminative power — documents that should be semantically distinct appear
    similar because their vectors all point in roughly the same direction.

    The score is computed as the mean pairwise cosine similarity across all
    vector pairs, using an efficient Gram matrix formulation:

        V_norm = L2_normalize(embeddings)       # shape (n, d)
        G      = V_norm @ V_norm.T              # shape (n, n)
        score  = (G.sum() - G.trace()) / (n * (n - 1))

    For large indices (n > sample_size), a random subsample is used for the
    Gram matrix computation to avoid O(n^2) memory and time costs. The
    eigenvalue distribution is always computed on the subsample.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (n, d). Any embedding model or dimension.
        Does not need to be pre-normalized.
    sample_size : int, optional
        Maximum number of vectors to use for Gram matrix computation.
        Defaults to 5000. Set to None to use all vectors (expensive for n > 10k).
    random_seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    AnisotropyResult
        Dataclass containing score, eigenvalues, interpretation, and metadata.

    Example
    -------
    >>> import numpy as np
    >>> from spectralyte.metrics.anisotropy import compute
    >>> embeddings = np.random.randn(1000, 384)
    >>> result = compute(embeddings)
    >>> print(result.score)          # near 0 for random vectors
    >>> print(result.interpretation) # 'healthy'
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D array of shape (n, d), got shape {embeddings.shape}"
        )

    n, d = embeddings.shape

    if n < 2:
        raise ValueError(f"Need at least 2 embeddings to compute anisotropy, got {n}")

    # ── Sampling ───────────────────────────────────────────────────────────────
    sampled = False
    effective_n = n

    if sample_size is not None and n > sample_size:
        rng = np.random.RandomState(random_seed)
        indices = rng.choice(n, size=sample_size, replace=False)
        working_embeddings = embeddings[indices]
        effective_n = sample_size
        sampled = True
    else:
        working_embeddings = embeddings

    # ── L2 normalize ──────────────────────────────────────────────────────────
    V = _l2_normalize(working_embeddings)   # shape (effective_n, d)

    # ── Gram matrix ───────────────────────────────────────────────────────────
    # G[i, j] = cosine_similarity(V[i], V[j]) since vectors are L2-normalized
    G = V @ V.T   # shape (effective_n, effective_n)

    # ── Anisotropy score ──────────────────────────────────────────────────────
    # Mean of all off-diagonal elements = mean pairwise cosine similarity
    # Off-diagonal sum = G.sum() - G.trace()
    # Number of off-diagonal pairs = n * (n - 1)
    off_diag_sum = G.sum() - np.trace(G)
    score = float(off_diag_sum / (effective_n * (effective_n - 1)))

    # Clamp to [0, 1] to handle floating point edge cases
    score = float(np.clip(score, 0.0, 1.0))

    # ── Eigenvalue distribution ───────────────────────────────────────────────
    # Eigenvalues of the Gram matrix reflect variance concentration.
    # Concentration in a few eigenvalues → anisotropic space.
    # Use eigh (symmetric matrix) for numerical stability and speed.
    eigenvalues = np.linalg.eigvalsh(G)
    eigenvalues = np.sort(eigenvalues)[::-1]   # descending order

    return AnisotropyResult(
        score=score,
        eigenvalues=eigenvalues,
        interpretation=_interpret(score),
        n_vectors=n,
        n_dims=d,
        sampled=sampled,
        sample_size=effective_n,
    )