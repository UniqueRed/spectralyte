"""
tests/test_metrics/test_dimensionality.py
==========================================
Tests for the effective dimensionality metric.

Key properties verified:
- Recovers correct effective dimensionality on low-rank synthetic data
- Participation ratio is bounded correctly
- Utilization is in [0, 1]
- Threshold parameter works correctly
- TruncatedSVD triggers for large inputs
- Input validation raises appropriate errors
- Interpretation thresholds map correctly
"""

import numpy as np
import pytest
from spectralyte.metrics.dimensionality import compute, DimensionalityResult


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def low_rank_embeddings():
    """
    Embeddings with known low effective dimensionality.
    Data lives on a 10-dimensional subspace of R^256.
    """
    rng = np.random.RandomState(42)
    # Create a rank-10 matrix: project 10-dim data into 256 dims
    basis = rng.randn(10, 256)       # 10 basis vectors in R^256
    coefficients = rng.randn(500, 10)  # 500 points in 10-dim space
    embeddings = coefficients @ basis  # shape (500, 256) but rank 10
    # Add tiny noise to make it numerically non-degenerate
    embeddings += rng.randn(500, 256) * 0.001
    return embeddings


@pytest.fixture
def full_rank_embeddings():
    """
    Embeddings using most of their nominal dimensionality.
    Random vectors in R^64 — effective dimensionality close to 64.
    """
    rng = np.random.RandomState(42)
    return rng.randn(500, 64)


# ── Core correctness tests ─────────────────────────────────────────────────────

def test_recovers_low_rank(low_rank_embeddings):
    """
    Effective dimensionality should be close to 10 for rank-10 data
    embedded in R^256.
    """
    result = compute(low_rank_embeddings, variance_threshold=0.99)
    assert result.effective_dims <= 20, (
        f"Expected effective_dims near 10 for rank-10 data, got {result.effective_dims}"
    )
    assert result.effective_dims <= result.nominal_dims


def test_full_rank_higher_than_low_rank(low_rank_embeddings, full_rank_embeddings):
    """Full-rank data should have higher effective dimensionality than low-rank."""
    low_result = compute(low_rank_embeddings)
    full_result = compute(full_rank_embeddings)
    # Utilization should be higher for full-rank data
    assert full_result.utilization > low_result.utilization, (
        f"Expected full-rank utilization ({full_result.utilization:.3f}) > "
        f"low-rank utilization ({low_result.utilization:.3f})"
    )


def test_effective_dims_leq_nominal(full_rank_embeddings):
    """Effective dims must never exceed nominal dims."""
    result = compute(full_rank_embeddings)
    assert result.effective_dims <= result.nominal_dims


def test_effective_dims_geq_one(full_rank_embeddings):
    """Effective dims must be at least 1."""
    result = compute(full_rank_embeddings)
    assert result.effective_dims >= 1


def test_threshold_95_less_than_99(full_rank_embeddings):
    """
    95% threshold should require fewer components than 99% threshold.
    """
    result_95 = compute(full_rank_embeddings, variance_threshold=0.95)
    result_99 = compute(full_rank_embeddings, variance_threshold=0.99)
    assert result_95.effective_dims <= result_99.effective_dims


def test_cumulative_variance_reaches_threshold(full_rank_embeddings):
    """
    Cumulative variance at effective_dims index must be >= threshold.
    """
    threshold = 0.95
    result = compute(full_rank_embeddings, variance_threshold=threshold)
    k = result.effective_dims
    # cumulative_variance is 0-indexed, so index k-1 is the k-th component
    assert result.cumulative_variance[k - 1] >= threshold - 1e-10, (
        f"Cumulative variance at k={k} is {result.cumulative_variance[k-1]:.4f}, "
        f"expected >= {threshold}"
    )


# ── Return type tests ──────────────────────────────────────────────────────────

def test_returns_dimensionality_result(full_rank_embeddings):
    """compute() must return a DimensionalityResult."""
    result = compute(full_rank_embeddings)
    assert isinstance(result, DimensionalityResult)


def test_result_has_all_fields(full_rank_embeddings):
    """All expected fields must be present."""
    result = compute(full_rank_embeddings)
    expected_fields = [
        'effective_dims', 'nominal_dims', 'utilization', 'participation_ratio',
        'explained_variance_ratio', 'cumulative_variance', 'variance_threshold',
        'interpretation', 'n_vectors', 'n_dims', 'truncated'
    ]
    for field in expected_fields:
        assert hasattr(result, field), f"Missing field: {field}"


def test_metadata_correct(full_rank_embeddings):
    """n_vectors, n_dims, nominal_dims must match input shape."""
    n, d = full_rank_embeddings.shape
    result = compute(full_rank_embeddings)
    assert result.n_vectors == n
    assert result.n_dims == d
    assert result.nominal_dims == d


def test_variance_threshold_stored(full_rank_embeddings):
    """variance_threshold must be stored in result."""
    threshold = 0.90
    result = compute(full_rank_embeddings, variance_threshold=threshold)
    assert result.variance_threshold == threshold


# ── Utilization tests ──────────────────────────────────────────────────────────

def test_utilization_bounded(full_rank_embeddings):
    """Utilization must be in (0, 1]."""
    result = compute(full_rank_embeddings)
    assert 0 < result.utilization <= 1.0


def test_utilization_equals_effective_over_nominal(full_rank_embeddings):
    """Utilization must equal effective_dims / nominal_dims."""
    result = compute(full_rank_embeddings)
    expected = result.effective_dims / result.nominal_dims
    assert abs(result.utilization - expected) < 1e-10


# ── Participation ratio tests ──────────────────────────────────────────────────

def test_participation_ratio_at_least_one(full_rank_embeddings):
    """Participation ratio must be >= 1."""
    result = compute(full_rank_embeddings)
    assert result.participation_ratio >= 1.0 - 1e-10


def test_participation_ratio_at_most_nominal_dims(full_rank_embeddings):
    """Participation ratio must be <= nominal_dims."""
    result = compute(full_rank_embeddings)
    assert result.participation_ratio <= result.nominal_dims + 1e-10


def test_participation_ratio_low_rank_is_small(low_rank_embeddings):
    """Low-rank data should have low participation ratio."""
    result = compute(low_rank_embeddings)
    # PR should be much less than nominal dims for low-rank data
    assert result.participation_ratio < result.nominal_dims * 0.3, (
        f"Expected low PR for rank-10 data in R^256, got {result.participation_ratio:.1f}"
    )


# ── Explained variance array tests ────────────────────────────────────────────

def test_explained_variance_sums_to_one(full_rank_embeddings):
    """Explained variance ratios must sum to approximately 1."""
    result = compute(full_rank_embeddings)
    total = result.explained_variance_ratio.sum()
    assert abs(total - 1.0) < 0.01, (
        f"Explained variance ratios sum to {total:.4f}, expected ~1.0"
    )


def test_explained_variance_non_negative(full_rank_embeddings):
    """All explained variance ratios must be non-negative."""
    result = compute(full_rank_embeddings)
    assert all(r >= -1e-10 for r in result.explained_variance_ratio)


def test_explained_variance_sorted_descending(full_rank_embeddings):
    """Explained variance ratios must be sorted descending."""
    result = compute(full_rank_embeddings)
    evr = result.explained_variance_ratio
    assert all(evr[i] >= evr[i+1] - 1e-10 for i in range(len(evr)-1))


def test_cumulative_variance_monotone(full_rank_embeddings):
    """Cumulative variance must be monotonically non-decreasing."""
    result = compute(full_rank_embeddings)
    cv = result.cumulative_variance
    assert all(cv[i] <= cv[i+1] + 1e-10 for i in range(len(cv)-1))


def test_cumulative_variance_ends_at_one(full_rank_embeddings):
    """Cumulative variance must end at approximately 1.0."""
    result = compute(full_rank_embeddings)
    assert abs(result.cumulative_variance[-1] - 1.0) < 0.01


# ── Interpretation tests ───────────────────────────────────────────────────────

def test_interpretation_is_valid_string(full_rank_embeddings):
    """Interpretation must be one of the four valid values."""
    result = compute(full_rank_embeddings)
    assert result.interpretation in {"healthy", "moderate", "low", "critical"}


def test_interpretation_boundaries():
    """Test interpretation threshold boundary values."""
    from spectralyte.metrics.dimensionality import _interpret
    assert _interpret(0.25) == "healthy"
    assert _interpret(0.20) == "healthy"
    assert _interpret(0.19) == "moderate"
    assert _interpret(0.10) == "moderate"
    assert _interpret(0.09) == "low"
    assert _interpret(0.04) == "low"
    assert _interpret(0.03) == "critical"
    assert _interpret(0.00) == "critical"


def test_low_rank_interpretation(low_rank_embeddings):
    """Low-rank data in high-dim space should not be 'healthy'."""
    result = compute(low_rank_embeddings)
    assert result.interpretation in {"moderate", "low", "critical"}, (
        f"Expected non-healthy interpretation for rank-10 data in R^256, "
        f"got '{result.interpretation}' (utilization={result.utilization:.3f})"
    )


# ── Dimension / size tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [32, 128, 384, 768])
def test_different_dimensions(d):
    """Should work on any embedding dimension."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, d)
    result = compute(embeddings)
    assert result.nominal_dims == d
    assert 1 <= result.effective_dims <= d


@pytest.mark.parametrize("n", [2, 10, 50, 200])
def test_different_sample_sizes(n):
    """Should work with any number of embeddings >= 2."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(n, 64)
    result = compute(embeddings)
    assert result.n_vectors == n


# ── TruncatedSVD tests ─────────────────────────────────────────────────────────

def test_truncated_svd_triggers():
    """TruncatedSVD should be used for large inputs."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(10_001, 64)
    result = compute(embeddings, truncation_threshold=10_000)
    assert result.truncated is True


def test_full_svd_for_small_inputs(full_rank_embeddings):
    """Full SVD should be used for small inputs."""
    result = compute(full_rank_embeddings, truncation_threshold=10_000)
    assert result.truncated is False


def test_truncated_and_full_give_similar_results():
    """
    TruncatedSVD and full SVD should give similar effective_dims
    on the same data.
    """
    rng = np.random.RandomState(42)
    embeddings = rng.randn(500, 64)
    full_result = compute(embeddings, truncation_threshold=10_000)
    trunc_result = compute(embeddings, truncation_threshold=0)   # force truncated
    # Effective dims should be the same or very close
    assert abs(full_result.effective_dims - trunc_result.effective_dims) <= 3


# ── Input validation tests ─────────────────────────────────────────────────────

def test_raises_for_1d_input():
    """1D input should raise ValueError."""
    with pytest.raises(ValueError, match="2D"):
        compute(np.array([1.0, 2.0, 3.0]))


def test_raises_for_single_vector():
    """Single vector (n=1) should raise ValueError."""
    with pytest.raises(ValueError, match="at least 2"):
        compute(np.array([[1.0, 2.0, 3.0]]))


def test_raises_for_invalid_threshold():
    """variance_threshold outside (0, 1] should raise ValueError."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 64)
    with pytest.raises(ValueError, match="variance_threshold"):
        compute(embeddings, variance_threshold=0.0)
    with pytest.raises(ValueError, match="variance_threshold"):
        compute(embeddings, variance_threshold=1.5)


def test_handles_identical_embeddings():
    """All identical embeddings should not crash."""
    base = np.ones((100, 64))
    result = compute(base)
    assert result is not None
    assert result.effective_dims >= 1


def test_variance_threshold_of_one():
    """variance_threshold=1.0 is valid and should be accepted."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 32)
    result = compute(embeddings, variance_threshold=1.0)
    assert result is not None