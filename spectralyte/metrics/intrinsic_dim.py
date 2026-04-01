"""
spectralyte/metrics/intrinsic_dim.py
======================================
Intrinsic dimensionality estimation via the TwoNN method.

Estimates the true geometric complexity of the data manifold — the minimum
number of dimensions needed to faithfully represent the relationships between
embeddings, independent of the nominal embedding dimension.

Unlike effective dimensionality (which measures how many dimensions carry
variance in the embedding space), intrinsic dimensionality measures the
fundamental complexity of the underlying data. A corpus of company FAQ
documents might have intrinsic dimensionality of 8 even when embedded in
1536 dimensions and even when effective dimensionality is 43 — because the
semantic relationships can be faithfully captured in 8 degrees of freedom.

Algorithm (TwoNN, Facco et al. 2017):
    For each point v_i, compute:
        mu_i = dist(v_i, v_{i,2}) / dist(v_i, v_{i,1})

    where v_{i,1} and v_{i,2} are the first and second nearest neighbors.

    Under the assumption that data lies on a d-dimensional manifold,
    the survival function of mu follows:
        P(mu > x) = x^(-d)

    Taking logs:
        log P(mu > x) = -d * log(x)

    Equivalently, plotting log(1 - F(mu)) against log(mu) gives a straight
    line with slope -d, where F(mu) is the empirical CDF of mu values.

    The intrinsic dimensionality d_int is estimated via ordinary least squares
    regression of log(1 - F(mu)) on log(mu).

    Boundary effects: points near the edge of the data distribution have
    distorted mu ratios. Standard practice trims the top trim_fraction of
    mu values before fitting.

Reference:
    Facco, E., d'Errico, M., Rodriguez, A., & Laio, A. (2017).
    Estimating the intrinsic dimension of datasets by a minimal neighborhood
    information. Scientific Reports, 7(1), 12140.
    arXiv:1706.05587
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.spatial import cKDTree
from scipy.stats import linregress


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class IntrinsicDimResult:
    """
    Result of a TwoNN intrinsic dimensionality estimation.

    Attributes
    ----------
    d_int : float
        Estimated intrinsic dimensionality. The slope of log(1-F(mu)) vs log(mu).
    r_squared : float
        R² of the linear fit. Measures goodness of fit.
        High R² (> 0.95) indicates clean manifold structure.
        Low R² indicates mixed dimensionality or noisy data.
    mu_values : np.ndarray
        Raw mu ratios for each embedding (second / first NN distance).
        Shape (n,) after trimming boundary points.
    log_mu : np.ndarray
        log(mu) values used in regression. Shape after trimming.
    log_survival : np.ndarray
        log(1 - empirical CDF of mu) values used in regression.
    slope : float
        Raw regression slope (negative of d_int).
    intercept : float
        Regression intercept (should be near 0 for well-behaved data).
    trim_fraction : float
        Fraction of high-mu values trimmed before fitting.
    n_points_used : int
        Number of mu values used in the regression after trimming.
    interpretation : str
        Human-readable assessment of the intrinsic dimensionality.
    n_vectors : int
        Total number of embedding vectors analyzed.
    n_dims : int
        Nominal embedding dimensionality.
    normalized : bool
        True if embeddings were L2-normalized before distance computation.
    """

    d_int: float
    r_squared: float
    mu_values: np.ndarray
    log_mu: np.ndarray
    log_survival: np.ndarray
    slope: float
    intercept: float
    trim_fraction: float
    n_points_used: int
    interpretation: str
    n_vectors: int
    n_dims: int
    normalized: bool


# ── Interpretation ─────────────────────────────────────────────────────────────

def _interpret(d_int: float, nominal_dims: int) -> str:
    """
    Assess intrinsic dimensionality relative to the nominal embedding dimension.

    Parameters
    ----------
    d_int : float
        Estimated intrinsic dimensionality.
    nominal_dims : int
        Nominal embedding dimensionality.

    Returns
    -------
    str
        One of 'low', 'moderate', 'high', 'very_high'.
    """
    ratio = d_int / nominal_dims
    if ratio < 0.05:
        return "low"
    elif ratio < 0.15:
        return "moderate"
    elif ratio < 0.30:
        return "high"
    else:
        return "very_high"


# ── Normalization ──────────────────────────────────────────────────────────────

def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows. Zero vectors are left unchanged."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


# ── TwoNN computation ──────────────────────────────────────────────────────────

def _compute_mu_values(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute mu_i = dist(v_i, v_{i,2}) / dist(v_i, v_{i,1}) for each point.

    The ratio of the second to first nearest neighbor distance is the core
    quantity in the TwoNN estimator. On a d-dimensional manifold, these ratios
    follow P(mu > x) = x^(-d).

    Points with identical nearest neighbors (distance 0) are excluded to
    avoid division by zero.

    Parameters
    ----------
    embeddings : np.ndarray
        L2-normalized embedding matrix of shape (n, d).

    Returns
    -------
    np.ndarray
        mu values for each valid point. May be shorter than n if any
        points have zero first-NN distance.
    """
    tree = cKDTree(embeddings)
    # Query 3 neighbors: index 0 is self (dist=0), 1 is first NN, 2 is second NN
    distances, _ = tree.query(embeddings, k=3)

    first_nn_dist = distances[:, 1]   # distance to nearest neighbor
    second_nn_dist = distances[:, 2]  # distance to second nearest neighbor

    # Exclude points where first NN distance is 0 (identical points)
    # to avoid division by zero
    valid_mask = first_nn_dist > 0
    first_valid = first_nn_dist[valid_mask]
    second_valid = second_nn_dist[valid_mask]

    mu = second_valid / first_valid
    return mu


def _fit_twonn(
    mu_values: np.ndarray,
    trim_fraction: float,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Fit the TwoNN model to mu values and estimate intrinsic dimensionality.

    Trims the top trim_fraction of mu values to reduce boundary effects,
    then fits a linear regression of log(1 - empirical_CDF) vs log(mu).
    The slope of this line is -d_int.

    Parameters
    ----------
    mu_values : np.ndarray
        Raw mu ratios (second NN / first NN distance).
    trim_fraction : float
        Fraction of high-mu values to trim before fitting.

    Returns
    -------
    Tuple of:
        d_int : float        - Estimated intrinsic dimensionality
        r_squared : float    - R² of linear fit
        slope : float        - Raw slope (negative of d_int)
        intercept : float    - Regression intercept
        log_mu : np.ndarray  - log(mu) values used in regression
        log_surv : np.ndarray - log(1 - F(mu)) values used in regression
    """
    # Sort mu values for empirical CDF computation
    mu_sorted = np.sort(mu_values)
    n = len(mu_sorted)

    # Trim top trim_fraction to remove boundary effects
    trim_n = int(n * (1 - trim_fraction))
    trim_n = max(10, trim_n)   # keep at least 10 points
    mu_trimmed = mu_sorted[:trim_n]

    # Empirical CDF: F(mu_i) = i / n (using full n for normalization)
    # This gives the fraction of points with mu <= mu_i
    ranks = np.arange(1, len(mu_trimmed) + 1)
    empirical_cdf = ranks / n

    # Survival function: 1 - F(mu)
    survival = 1.0 - empirical_cdf

    # Avoid log(0) — filter out any zero survival values
    valid = survival > 0
    mu_fit = mu_trimmed[valid]
    survival_fit = survival[valid]

    if len(mu_fit) < 5:
        # Degenerate case — not enough points for meaningful fit
        return 1.0, 0.0, -1.0, 0.0, np.log(mu_fit), np.log(survival_fit)

    log_mu = np.log(mu_fit)
    log_surv = np.log(survival_fit)

    # Linear regression: log(1 - F) = slope * log(mu) + intercept
    # Under TwoNN model, slope should equal -d_int
    result = linregress(log_mu, log_surv)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r_squared = float(result.rvalue ** 2)

    # d_int = -slope (slope is negative for valid manifold data)
    d_int = max(1.0, -slope)   # clamp to >= 1 (can't have < 1D manifold)

    return d_int, r_squared, slope, intercept, log_mu, log_surv


# ── Core computation ───────────────────────────────────────────────────────────

def compute(
    embeddings: np.ndarray,
    trim_fraction: float = 0.10,
    sample_size: Optional[int] = 5000,
    random_seed: int = 42,
    normalize: Optional[bool] = None,
) -> IntrinsicDimResult:
    """
    Estimate the intrinsic dimensionality of an embedding space using TwoNN.

    The TwoNN estimator (Facco et al. 2017) estimates the minimum number of
    dimensions needed to faithfully represent the geometric structure of the
    data, independent of the nominal embedding dimension. It requires only
    the ratio of each point's second to first nearest neighbor distance.

    Under the assumption that data lies on a d-dimensional manifold, the
    distribution of these ratios follows P(mu > x) = x^(-d). The intrinsic
    dimensionality is estimated as the negative slope of a log-log plot of
    the survival function of mu.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (n, d). Does not need to be pre-normalized.
    trim_fraction : float
        Fraction of highest mu values to trim before fitting. Default 0.10.
        Trimming reduces boundary effects where the manifold assumption breaks.
    sample_size : Optional[int]
        Maximum number of embeddings for mu computation. Default 5000.
        Set to None to use all embeddings.
    random_seed : int
        Random seed for reproducible sampling.
    normalize : Optional[bool]
        Whether to L2-normalize before distance computation. If None,
        normalizes for d > 100 (consistent with density metric). Default None.

    Returns
    -------
    IntrinsicDimResult
        Dataclass with d_int, r_squared, mu_values, fit data, interpretation,
        and metadata.

    Raises
    ------
    ValueError
        If embeddings is not 2D, trim_fraction not in [0, 1), or n < 3.

    Example
    -------
    >>> import numpy as np
    >>> from spectralyte.metrics.intrinsic_dim import compute
    >>> # 2D manifold (circle) embedded in R^100
    >>> theta = np.linspace(0, 2 * np.pi, 500)
    >>> circle = np.column_stack([np.cos(theta), np.sin(theta)])
    >>> embeddings = np.pad(circle, ((0, 0), (0, 98)))   # embed in R^100
    >>> result = compute(embeddings)
    >>> print(f"Intrinsic dim: {result.d_int:.2f}")   # should be near 1
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D array of shape (n, d), got shape {embeddings.shape}"
        )
    if not (0.0 <= trim_fraction < 1.0):
        raise ValueError(
            f"trim_fraction must be in [0, 1), got {trim_fraction}"
        )

    n, d = embeddings.shape

    if n < 3:
        raise ValueError(
            f"TwoNN requires at least 3 embeddings (need 2 distinct neighbors), got {n}"
        )

    # ── Normalization decision ─────────────────────────────────────────────────
    if normalize is None:
        should_normalize = d > 100
    else:
        should_normalize = normalize

    V = _l2_normalize(embeddings) if should_normalize else embeddings.copy()

    # ── Sampling ───────────────────────────────────────────────────────────────
    if sample_size is not None and n > sample_size:
        rng = np.random.RandomState(random_seed)
        indices = rng.choice(n, size=sample_size, replace=False)
        V_sample = V[indices]
    else:
        V_sample = V

    # ── Compute mu values ──────────────────────────────────────────────────────
    mu_values = _compute_mu_values(V_sample)

    if len(mu_values) < 5:
        raise ValueError(
            f"Too few valid mu values ({len(mu_values)}) after filtering. "
            f"Check for duplicate embeddings in your index."
        )

    # ── Fit TwoNN model ────────────────────────────────────────────────────────
    d_int, r_squared, slope, intercept, log_mu, log_surv = _fit_twonn(
        mu_values, trim_fraction
    )

    return IntrinsicDimResult(
        d_int=d_int,
        r_squared=r_squared,
        mu_values=mu_values,
        log_mu=log_mu,
        log_survival=log_surv,
        slope=slope,
        intercept=intercept,
        trim_fraction=trim_fraction,
        n_points_used=len(log_mu),
        interpretation=_interpret(d_int, d),
        n_vectors=n,
        n_dims=d,
        normalized=should_normalize,
    )