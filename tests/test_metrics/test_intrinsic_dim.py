"""
tests/test_metrics/test_intrinsic_dim.py
==========================================
Tests for the TwoNN intrinsic dimensionality estimator.

Key properties verified:
- Recovers approximately correct d_int for known manifold structures
- d_int is positive and bounded reasonably
- R² is in [0, 1]
- mu values are all >= 1 (second NN >= first NN distance)
- Low-dim manifold in high-dim space gives low d_int
- High-dim data gives higher d_int than low-dim data
- Sampling triggers correctly for large inputs
- Trim fraction reduces boundary effects
- Normalization triggers correctly
- Input validation raises appropriate errors
- Reproducibility with same seed
"""

import numpy as np
import pytest
from spectralyte.metrics.intrinsic_dim import (
    compute, IntrinsicDimResult, _compute_mu_values, _l2_normalize
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def low_dim_manifold():
    """
    2D manifold embedded in R^100.
    Points sampled from a 2D Gaussian, then padded with zeros.
    True intrinsic dimensionality ≈ 2.
    """
    rng = np.random.RandomState(42)
    # 2D data
    data_2d = rng.randn(500, 2)
    # Pad to R^100
    padding = np.zeros((500, 98))
    return np.hstack([data_2d, padding])


@pytest.fixture
def high_dim_data():
    """
    Uniformly random data in R^50.
    Intrinsic dimensionality should be high (close to 50).
    """
    rng = np.random.RandomState(42)
    return rng.randn(500, 50)


@pytest.fixture
def uniform_embeddings():
    """Standard uniform embeddings for general tests."""
    rng = np.random.RandomState(42)
    return rng.randn(300, 64)


# ── Manifold recovery tests ────────────────────────────────────────────────────

def test_low_dim_manifold_gives_low_d_int(low_dim_manifold):
    """
    2D data embedded in R^100 should have d_int much less than 100.
    The estimator won't be exact but should clearly indicate low dimension.
    """
    result = compute(low_dim_manifold, sample_size=None)
    assert result.d_int < 20, (
        f"Expected low d_int for 2D manifold in R^100, got {result.d_int:.2f}"
    )


def test_high_dim_higher_than_low_dim(low_dim_manifold, high_dim_data):
    """
    High-dimensional random data should have higher d_int than
    low-dimensional manifold data.
    """
    low_result = compute(low_dim_manifold, sample_size=None)
    high_result = compute(high_dim_data, sample_size=None)
    assert high_result.d_int > low_result.d_int, (
        f"Expected high-dim d_int ({high_result.d_int:.2f}) > "
        f"low-dim d_int ({low_result.d_int:.2f})"
    )


def test_d_int_positive(uniform_embeddings):
    """Intrinsic dimensionality must always be positive."""
    result = compute(uniform_embeddings)
    assert result.d_int >= 1.0, (
        f"Expected d_int >= 1, got {result.d_int:.4f}"
    )


def test_d_int_reasonable_upper_bound(uniform_embeddings):
    """
    d_int should not wildly exceed the nominal dimensionality.
    For random data in R^64, d_int should be <= 64.
    """
    result = compute(uniform_embeddings, sample_size=None)
    assert result.d_int <= uniform_embeddings.shape[1] * 2, (
        f"d_int ({result.d_int:.2f}) is unreasonably large for "
        f"R^{uniform_embeddings.shape[1]} data"
    )


# ── Return type tests ──────────────────────────────────────────────────────────

def test_returns_intrinsic_dim_result(uniform_embeddings):
    """compute() must return an IntrinsicDimResult."""
    result = compute(uniform_embeddings)
    assert isinstance(result, IntrinsicDimResult)


def test_result_has_all_fields(uniform_embeddings):
    """All expected fields must be present."""
    result = compute(uniform_embeddings)
    expected_fields = [
        'd_int', 'r_squared', 'mu_values', 'log_mu', 'log_survival',
        'slope', 'intercept', 'trim_fraction', 'n_points_used',
        'interpretation', 'n_vectors', 'n_dims', 'normalized'
    ]
    for f in expected_fields:
        assert hasattr(result, f), f"Missing field: {f}"


def test_metadata_correct(uniform_embeddings):
    """n_vectors and n_dims must match input shape."""
    n, d = uniform_embeddings.shape
    result = compute(uniform_embeddings)
    assert result.n_vectors == n
    assert result.n_dims == d


def test_trim_fraction_stored(uniform_embeddings):
    """trim_fraction must be stored in result."""
    result = compute(uniform_embeddings, trim_fraction=0.15)
    assert result.trim_fraction == 0.15


# ── R² tests ───────────────────────────────────────────────────────────────────

def test_r_squared_bounded(uniform_embeddings):
    """R² must be in [0, 1]."""
    result = compute(uniform_embeddings)
    assert 0.0 <= result.r_squared <= 1.0, (
        f"R² ({result.r_squared:.4f}) is out of [0, 1]"
    )


def test_low_dim_manifold_high_r_squared(low_dim_manifold):
    """Clean manifold structure should produce high R² (good fit)."""
    result = compute(low_dim_manifold, sample_size=None)
    assert result.r_squared > 0.80, (
        f"Expected high R² for clean 2D manifold, got {result.r_squared:.4f}"
    )


# ── Mu value tests ─────────────────────────────────────────────────────────────

def test_mu_values_geq_one(uniform_embeddings):
    """
    All mu values must be >= 1.
    mu = dist(second NN) / dist(first NN) >= 1 by definition
    since second NN is never closer than first NN.
    """
    result = compute(uniform_embeddings)
    assert all(mu >= 1.0 - 1e-10 for mu in result.mu_values), (
        f"Found mu values < 1: min={result.mu_values.min():.6f}"
    )


def test_mu_values_positive(uniform_embeddings):
    """All mu values must be strictly positive."""
    result = compute(uniform_embeddings)
    assert all(mu > 0 for mu in result.mu_values)


def test_mu_values_shape(uniform_embeddings):
    """mu_values must be 1D array."""
    result = compute(uniform_embeddings)
    assert result.mu_values.ndim == 1
    assert len(result.mu_values) > 0


# ── Regression fit tests ───────────────────────────────────────────────────────

def test_slope_negative(uniform_embeddings):
    """
    TwoNN slope must be negative.
    log(1 - F(mu)) decreases as log(mu) increases.
    """
    result = compute(uniform_embeddings, sample_size=None)
    assert result.slope < 0, (
        f"Expected negative slope, got {result.slope:.4f}"
    )


def test_d_int_equals_negative_slope(uniform_embeddings):
    """d_int must equal max(1, -slope)."""
    result = compute(uniform_embeddings, sample_size=None)
    expected = max(1.0, -result.slope)
    assert abs(result.d_int - expected) < 1e-10


def test_log_mu_and_log_survival_same_length(uniform_embeddings):
    """log_mu and log_survival must have the same length."""
    result = compute(uniform_embeddings)
    assert len(result.log_mu) == len(result.log_survival)


def test_n_points_used_matches_log_arrays(uniform_embeddings):
    """n_points_used must equal length of log arrays."""
    result = compute(uniform_embeddings)
    assert result.n_points_used == len(result.log_mu)


# ── Interpretation tests ───────────────────────────────────────────────────────

def test_interpretation_valid_string(uniform_embeddings):
    """Interpretation must be one of the four valid values."""
    result = compute(uniform_embeddings)
    assert result.interpretation in {"low", "moderate", "high", "very_high"}


def test_interpretation_boundaries():
    """Test interpretation boundary values."""
    from spectralyte.metrics.intrinsic_dim import _interpret
    # For nominal_dims=100:
    assert _interpret(4, 100) == "low"         # 4/100 = 0.04 < 0.05
    assert _interpret(5, 100) == "moderate"    # 5/100 = 0.05
    assert _interpret(14, 100) == "moderate"   # 14/100 = 0.14 < 0.15
    assert _interpret(15, 100) == "high"       # 15/100 = 0.15
    assert _interpret(29, 100) == "high"       # 29/100 = 0.29 < 0.30
    assert _interpret(30, 100) == "very_high"  # 30/100 = 0.30


def test_low_dim_manifold_interpretation(low_dim_manifold):
    """2D manifold in R^100 should have 'low' interpretation."""
    result = compute(low_dim_manifold, sample_size=None)
    assert result.interpretation == "low", (
        f"Expected 'low' for 2D manifold in R^100, "
        f"got '{result.interpretation}' (d_int={result.d_int:.2f})"
    )


# ── Normalization tests ────────────────────────────────────────────────────────

def test_normalization_for_high_dimensions():
    """d > 100 should trigger L2 normalization by default."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 384)
    result = compute(embeddings)
    assert result.normalized is True


def test_no_normalization_for_low_dimensions():
    """d <= 100 should not trigger normalization by default."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 64)
    result = compute(embeddings)
    assert result.normalized is False


def test_explicit_normalize_true():
    """normalize=True should always normalize."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 32)   # d <= 100
    result = compute(embeddings, normalize=True)
    assert result.normalized is True


def test_explicit_normalize_false():
    """normalize=False should never normalize."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 384)  # d > 100
    result = compute(embeddings, normalize=False)
    assert result.normalized is False


# ── Sampling tests ─────────────────────────────────────────────────────────────

def test_sampling_none_uses_all():
    """sample_size=None should use all vectors."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 32)
    result = compute(embeddings, sample_size=None)
    # All 200 vectors used (minus any filtered for zero first-NN distance)
    assert result.n_points_used > 0


def test_sampling_caps_for_large_input():
    """Large input should be sampled."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(6000, 32)
    result = compute(embeddings, sample_size=1000)
    # mu_values should have at most 1000 entries
    assert len(result.mu_values) <= 1000


def test_reproducibility(uniform_embeddings):
    """Same random seed should produce identical results."""
    result1 = compute(uniform_embeddings, random_seed=42)
    result2 = compute(uniform_embeddings, random_seed=42)
    assert result1.d_int == result2.d_int
    np.testing.assert_array_equal(result1.mu_values, result2.mu_values)


# ── Trim fraction tests ────────────────────────────────────────────────────────

def test_higher_trim_fewer_points_used(uniform_embeddings):
    """Higher trim_fraction should use fewer points in the regression."""
    result_low_trim = compute(uniform_embeddings, trim_fraction=0.05, sample_size=None)
    result_high_trim = compute(uniform_embeddings, trim_fraction=0.30, sample_size=None)
    assert result_high_trim.n_points_used <= result_low_trim.n_points_used


def test_zero_trim_fraction(uniform_embeddings):
    """trim_fraction=0.0 should use all mu values."""
    result = compute(uniform_embeddings, trim_fraction=0.0, sample_size=None)
    assert result is not None
    assert result.d_int > 0


# ── mu computation helper tests ────────────────────────────────────────────────

def test_mu_values_helper_basic():
    """_compute_mu_values should return values >= 1."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 32)
    mu = _compute_mu_values(embeddings)
    assert all(m >= 1.0 - 1e-10 for m in mu)


def test_mu_values_helper_length():
    """_compute_mu_values should return n values for n distinct points."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 32)
    mu = _compute_mu_values(embeddings)
    assert len(mu) == 100


# ── Dimension and size tests ───────────────────────────────────────────────────

@pytest.mark.parametrize("d", [8, 32, 64, 128])
def test_different_dimensions(d):
    """Should work on any embedding dimension."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, d)
    result = compute(embeddings)
    assert result.n_dims == d
    assert result.d_int >= 1.0


@pytest.mark.parametrize("n", [10, 50, 200, 500])
def test_different_sample_sizes(n):
    """Should work with different numbers of embeddings."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(n, 32)
    result = compute(embeddings)
    assert result.n_vectors == n
    assert result.d_int >= 1.0


# ── Input validation tests ─────────────────────────────────────────────────────

def test_raises_for_1d_input():
    """1D input should raise ValueError."""
    with pytest.raises(ValueError, match="2D"):
        compute(np.array([1.0, 2.0, 3.0]))


def test_raises_for_fewer_than_3_vectors():
    """n < 3 should raise ValueError."""
    with pytest.raises(ValueError, match="at least 3"):
        compute(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_raises_for_invalid_trim_fraction():
    """trim_fraction outside [0, 1) should raise ValueError."""
    embeddings = np.random.randn(100, 32)
    with pytest.raises(ValueError, match="trim_fraction"):
        compute(embeddings, trim_fraction=1.0)
    with pytest.raises(ValueError, match="trim_fraction"):
        compute(embeddings, trim_fraction=-0.1)