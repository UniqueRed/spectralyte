"""
tests/test_metrics/test_density.py
====================================
Tests for the density distribution metric.

Key properties verified:
- Low CV for uniform distributions
- High CV for clustered distributions
- Clustered space has higher CV than uniform space
- LOF scores are non-negative
- Outlier detection identifies isolated points
- Normalization triggers correctly for high dimensions
- k-NN distances are non-negative
- Sampling triggers for large inputs
- Input validation raises appropriate errors
- Interpretation thresholds map correctly
"""

import numpy as np
import pytest
from spectralyte.metrics.density import compute, DensityResult


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def uniform_embeddings():
    """
    Uniformly distributed embeddings — low CV expected.
    """
    rng = np.random.RandomState(42)
    return rng.randn(400, 64)


@pytest.fixture
def clustered_embeddings():
    """
    Embeddings in 5 clusters with highly varied internal densities.
    Some clusters are very tight, others are loose — producing high CV.
    """
    rng = np.random.RandomState(42)
    centers = rng.randn(5, 64) * 20   # well-separated centers
    clusters = []
    # Vary cluster sizes and spreads dramatically to produce high CV
    spreads = [0.05, 0.1, 0.5, 1.0, 3.0]   # very different densities
    sizes = [20, 40, 60, 100, 180]           # different cluster sizes
    for center, spread, size in zip(centers, spreads, sizes):
        cluster = center + rng.randn(size, 64) * spread
        clusters.append(cluster)
    return np.vstack(clusters)


@pytest.fixture
def embeddings_with_outlier():
    """
    Uniform embeddings with one obvious isolated outlier.
    """
    rng = np.random.RandomState(42)
    base = rng.randn(200, 32)
    # Add one point far from everything else
    outlier = np.array([[100.0] * 32])
    return np.vstack([base, outlier])   # shape (201, 32)


# ── Core correctness tests ─────────────────────────────────────────────────────

def test_uniform_has_low_cv(uniform_embeddings):
    """Uniformly distributed embeddings should have low CV."""
    result = compute(uniform_embeddings)
    assert result.cv < 0.6, (
        f"Expected low CV for uniform embeddings, got {result.cv:.4f}"
    )


def test_clustered_has_high_cv(clustered_embeddings):
    """Clustered embeddings should have high CV."""
    result = compute(clustered_embeddings)
    assert result.cv > 0.5, (
        f"Expected high CV for clustered embeddings, got {result.cv:.4f}"
    )


def test_clustered_cv_greater_than_uniform(uniform_embeddings, clustered_embeddings):
    """Clustered space must have strictly higher CV than uniform space."""
    uniform_result = compute(uniform_embeddings)
    clustered_result = compute(clustered_embeddings)
    assert clustered_result.cv > uniform_result.cv, (
        f"Expected clustered CV ({clustered_result.cv:.4f}) > "
        f"uniform CV ({uniform_result.cv:.4f})"
    )


def test_outlier_detected(embeddings_with_outlier):
    """The isolated outlier point should be detected with high LOF score."""
    result = compute(embeddings_with_outlier, lof_threshold=1.5)
    # The last point (index 200) is the outlier — it should be in outlier_indices
    assert 200 in result.outlier_indices, (
        f"Expected isolated point at index 200 to be detected as outlier. "
        f"Outliers found at: {result.outlier_indices}"
    )


# ── Return type tests ──────────────────────────────────────────────────────────

def test_returns_density_result(uniform_embeddings):
    """compute() must return a DensityResult."""
    result = compute(uniform_embeddings)
    assert isinstance(result, DensityResult)


def test_result_has_all_fields(uniform_embeddings):
    """All expected fields must be present."""
    result = compute(uniform_embeddings)
    expected_fields = [
        'cv', 'mean_knn_distance', 'std_knn_distance', 'knn_distances',
        'lof_scores', 'n_outliers', 'outlier_indices', 'interpretation',
        'k', 'lof_threshold', 'n_vectors', 'n_dims', 'normalized'
    ]
    for f in expected_fields:
        assert hasattr(result, f), f"Missing field: {f}"


def test_metadata_correct(uniform_embeddings):
    """n_vectors and n_dims must match input shape."""
    n, d = uniform_embeddings.shape
    result = compute(uniform_embeddings)
    assert result.n_vectors == n
    assert result.n_dims == d


def test_k_stored(uniform_embeddings):
    """k parameter must be stored in result."""
    result = compute(uniform_embeddings, k=5)
    assert result.k == 5


def test_lof_threshold_stored(uniform_embeddings):
    """lof_threshold must be stored in result."""
    result = compute(uniform_embeddings, lof_threshold=2.0)
    assert result.lof_threshold == 2.0


# ── k-NN distance tests ────────────────────────────────────────────────────────

def test_knn_distances_shape(uniform_embeddings):
    """k-NN distances must have shape (n,)."""
    n = len(uniform_embeddings)
    result = compute(uniform_embeddings)
    assert result.knn_distances.shape == (n,)


def test_knn_distances_non_negative(uniform_embeddings):
    """All k-NN distances must be non-negative."""
    result = compute(uniform_embeddings)
    assert all(d >= 0 for d in result.knn_distances)


def test_mean_knn_distance_positive(uniform_embeddings):
    """Mean k-NN distance must be positive for non-degenerate data."""
    result = compute(uniform_embeddings)
    assert result.mean_knn_distance > 0


def test_std_knn_distance_non_negative(uniform_embeddings):
    """Std of k-NN distances must be non-negative."""
    result = compute(uniform_embeddings)
    assert result.std_knn_distance >= 0


def test_cv_equals_std_over_mean(uniform_embeddings):
    """CV must equal std / mean of k-NN distances."""
    result = compute(uniform_embeddings)
    expected_cv = result.std_knn_distance / result.mean_knn_distance
    assert abs(result.cv - expected_cv) < 1e-10


# ── LOF score tests ────────────────────────────────────────────────────────────

def test_lof_scores_shape(uniform_embeddings):
    """LOF scores must have shape (n,)."""
    n = len(uniform_embeddings)
    result = compute(uniform_embeddings)
    assert result.lof_scores.shape == (n,)


def test_lof_scores_non_negative(uniform_embeddings):
    """All LOF scores must be non-negative."""
    result = compute(uniform_embeddings)
    assert all(s >= 0 for s in result.lof_scores)


def test_outlier_indices_consistent_with_scores(uniform_embeddings):
    """outlier_indices must match where lof_scores > lof_threshold."""
    result = compute(uniform_embeddings, lof_threshold=1.5)
    expected_outliers = set(np.where(result.lof_scores > 1.5)[0])
    actual_outliers = set(result.outlier_indices)
    assert expected_outliers == actual_outliers


def test_n_outliers_consistent(uniform_embeddings):
    """n_outliers must equal len(outlier_indices)."""
    result = compute(uniform_embeddings)
    assert result.n_outliers == len(result.outlier_indices)


def test_higher_threshold_fewer_outliers(uniform_embeddings):
    """Higher LOF threshold should produce fewer or equal outliers."""
    result_low = compute(uniform_embeddings, lof_threshold=1.2)
    result_high = compute(uniform_embeddings, lof_threshold=2.0)
    assert result_high.n_outliers <= result_low.n_outliers


# ── Normalization tests ────────────────────────────────────────────────────────

def test_normalization_for_high_dimensions():
    """d > 100 should trigger L2 normalization."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 384)
    result = compute(embeddings)
    assert result.normalized is True


def test_no_normalization_for_low_dimensions():
    """d <= 100 should not trigger L2 normalization."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 64)
    result = compute(embeddings)
    assert result.normalized is False


def test_normalization_boundary():
    """d = 100 should not trigger normalization (threshold is d > 100)."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 100)
    result = compute(embeddings)
    assert result.normalized is False


# ── Interpretation tests ───────────────────────────────────────────────────────

def test_interpretation_valid_string(uniform_embeddings):
    """Interpretation must be one of the four valid values."""
    result = compute(uniform_embeddings)
    assert result.interpretation in {"uniform", "moderate", "clustered", "severe"}


def test_interpretation_boundaries():
    """Test interpretation threshold boundary values."""
    from spectralyte.metrics.density import _interpret
    assert _interpret(0.0) == "uniform"
    assert _interpret(0.29) == "uniform"
    assert _interpret(0.3) == "moderate"
    assert _interpret(0.59) == "moderate"
    assert _interpret(0.6) == "clustered"
    assert _interpret(0.99) == "clustered"
    assert _interpret(1.0) == "severe"
    assert _interpret(2.0) == "severe"


def test_clustered_interpretation(clustered_embeddings):
    """Strongly clustered embeddings should not be 'uniform'."""
    result = compute(clustered_embeddings)
    assert result.interpretation != "uniform", (
        f"Expected non-uniform interpretation for clustered data, "
        f"got '{result.interpretation}' (CV={result.cv:.3f})"
    )


# ── Sampling tests ─────────────────────────────────────────────────────────────

def test_large_input_completes():
    """Should complete without error for large inputs."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(15_000, 64)
    result = compute(embeddings, sample_size=10_000)
    assert result is not None
    assert result.n_vectors == 15_000


def test_lof_scores_length_matches_full_index():
    """LOF scores must have length n even when sampling is used."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(500, 32)
    result = compute(embeddings, sample_size=200)
    assert len(result.lof_scores) == 500


def test_no_sampling_when_sample_size_none(uniform_embeddings):
    """sample_size=None should use all vectors for LOF."""
    result = compute(uniform_embeddings, sample_size=None)
    assert result is not None
    assert len(result.lof_scores) == len(uniform_embeddings)


# ── Dimension / size tests ─────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [16, 64, 128, 384])
def test_different_dimensions(d):
    """Should work on any embedding dimension."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, d)
    result = compute(embeddings)
    assert result.n_dims == d
    assert result.cv >= 0


@pytest.mark.parametrize("k", [3, 5, 10, 20])
def test_different_k_values(k):
    """Should work with different values of k."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 32)
    result = compute(embeddings, k=k)
    assert result.k == k
    assert len(result.knn_distances) == 200


# ── Input validation tests ─────────────────────────────────────────────────────

def test_raises_for_1d_input():
    """1D input should raise ValueError."""
    with pytest.raises(ValueError, match="2D"):
        compute(np.array([1.0, 2.0, 3.0]))


def test_raises_when_k_geq_n():
    """k >= n should raise ValueError."""
    embeddings = np.random.randn(10, 32)
    with pytest.raises(ValueError):
        compute(embeddings, k=10)


def test_raises_when_k_zero():
    """k = 0 should raise ValueError."""
    embeddings = np.random.randn(100, 32)
    with pytest.raises(ValueError):
        compute(embeddings, k=0)


def test_handles_zero_vectors():
    """Zero vectors should not cause division by zero."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 64)
    embeddings[0] = 0.0
    result = compute(embeddings)
    assert result is not None
    assert not np.isnan(result.cv)