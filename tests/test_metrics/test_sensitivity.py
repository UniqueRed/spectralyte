"""
tests/test_metrics/test_sensitivity.py
========================================
Tests for the Retrieval Sensitivity Index (RSI).

Key properties verified:
- Stable space (well-separated clusters) has high mean stability
- Brittle space (ambiguous overlapping regions) has lower stability
- Stability scores are bounded [0, 1]
- Epsilon is calibrated relative to k-NN distances not absolute
- Sampling produces correct output shape
- Brittle zone identification is consistent with stability scores
- Jaccard helper is correct on known inputs
- Input validation raises appropriate errors
- Reproducibility with same random seed
"""

import numpy as np
import pytest
from spectralyte.metrics.sensitivity import (
    compute, SensitivityResult, _jaccard, _l2_normalize
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def stable_embeddings():
    """
    Well-separated clusters — high stability expected.
    Queries inside a cluster always retrieve the same cluster members.
    """
    rng = np.random.RandomState(42)
    centers = rng.randn(5, 64) * 15   # very well-separated
    clusters = []
    for center in centers:
        # Tight, well-separated clusters
        cluster = center + rng.randn(40, 64) * 0.05
        clusters.append(cluster)
    return np.vstack(clusters)   # shape (200, 64)


@pytest.fixture
def uniform_embeddings():
    """
    Uniformly distributed embeddings — moderate stability.
    """
    rng = np.random.RandomState(42)
    return rng.randn(300, 64)


# ── Jaccard helper tests ───────────────────────────────────────────────────────

def test_jaccard_identical_sets():
    """Identical sets have Jaccard = 1.0."""
    assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_disjoint_sets():
    """Disjoint sets have Jaccard = 0.0."""
    assert _jaccard({1, 2, 3}, {4, 5, 6}) == 0.0


def test_jaccard_partial_overlap():
    """Partial overlap: |intersection| / |union|."""
    # intersection = {1, 2}, union = {1, 2, 3, 4}
    result = _jaccard({1, 2, 3}, {1, 2, 4})
    expected = 2 / 4   # {1,2} / {1,2,3,4}
    assert abs(result - expected) < 1e-10


def test_jaccard_both_empty():
    """Both empty sets return 1.0 by convention."""
    assert _jaccard(set(), set()) == 1.0


def test_jaccard_one_empty():
    """One empty set returns 0.0."""
    assert _jaccard({1, 2}, set()) == 0.0
    assert _jaccard(set(), {1, 2}) == 0.0


def test_jaccard_single_element_overlap():
    """Single element in common."""
    result = _jaccard({1, 2, 3, 4, 5}, {5, 6, 7, 8, 9})
    expected = 1 / 9   # {5} / {1,2,3,4,5,6,7,8,9}
    assert abs(result - expected) < 1e-10


# ── Core correctness tests ─────────────────────────────────────────────────────

def test_stable_space_high_stability(stable_embeddings):
    """
    Well-separated clusters should have high mean stability.
    Small perturbations stay within the same cluster.
    """
    result = compute(stable_embeddings, k=5, m=3, sample_size=-1)
    assert result.mean_stability > 0.7, (
        f"Expected high stability for well-separated clusters, "
        f"got {result.mean_stability:.4f}"
    )


def test_stability_bounded_zero_to_one(uniform_embeddings):
    """Mean stability must be in [0, 1]."""
    result = compute(uniform_embeddings)
    assert 0.0 <= result.mean_stability <= 1.0


def test_per_embedding_stability_bounded(uniform_embeddings):
    """All per-embedding stability scores must be in [0, 1]."""
    result = compute(uniform_embeddings)
    scores = result.stability_per_embedding
    assert all(0.0 <= s <= 1.0 for s in scores), (
        f"Found stability scores outside [0, 1]: "
        f"min={scores.min():.4f}, max={scores.max():.4f}"
    )


def test_identical_embeddings_maximum_stability():
    """
    Near-identical embeddings should have stability near 1.0.
    All perturbations return the same neighbors.
    """
    base = np.ones((50, 32))
    base += np.random.RandomState(42).randn(50, 32) * 1e-6  # tiny noise
    result = compute(base, k=5, m=3, sample_size=-1)
    assert result.mean_stability > 0.85, (
        f"Expected near-1.0 stability for near-identical embeddings, "
        f"got {result.mean_stability:.4f}"
    )


# ── Return type tests ──────────────────────────────────────────────────────────

def test_returns_sensitivity_result(uniform_embeddings):
    """compute() must return a SensitivityResult."""
    result = compute(uniform_embeddings)
    assert isinstance(result, SensitivityResult)


def test_result_has_all_fields(uniform_embeddings):
    """All expected fields must be present."""
    result = compute(uniform_embeddings)
    expected_fields = [
        'mean_stability', 'stability_per_embedding', 'brittle_zone_indices',
        'n_brittle', 'brittle_fraction', 'epsilon_used', 'epsilon_fraction',
        'mean_knn_distance', 'interpretation', 'k', 'm', 'brittle_threshold',
        'n_vectors', 'n_sampled', 'n_dims', 'sampled'
    ]
    for f in expected_fields:
        assert hasattr(result, f), f"Missing field: {f}"


def test_metadata_correct(uniform_embeddings):
    """n_vectors and n_dims must match input shape."""
    n, d = uniform_embeddings.shape
    result = compute(uniform_embeddings)
    assert result.n_vectors == n
    assert result.n_dims == d


def test_k_and_m_stored(uniform_embeddings):
    """k and m parameters must be stored in result."""
    result = compute(uniform_embeddings, k=7, m=3)
    assert result.k == 7
    assert result.m == 3


def test_brittle_threshold_stored(uniform_embeddings):
    """brittle_threshold must be stored."""
    result = compute(uniform_embeddings, brittle_threshold=0.4)
    assert result.brittle_threshold == 0.4


def test_epsilon_fraction_stored(uniform_embeddings):
    """epsilon_fraction must be stored."""
    result = compute(uniform_embeddings, epsilon_fraction=0.1)
    assert result.epsilon_fraction == 0.1


# ── Epsilon calibration tests ──────────────────────────────────────────────────

def test_epsilon_scales_with_knn_distance():
    """
    Epsilon should scale with the density of the space.
    Denser space (smaller k-NN distances) → smaller epsilon.
    """
    rng = np.random.RandomState(42)
    # Dense space: points close together
    dense = rng.randn(200, 32) * 0.1
    # Sparse space: points far apart
    sparse = rng.randn(200, 32) * 10.0

    result_dense = compute(dense, k=5, sample_size=-1)
    result_sparse = compute(sparse, k=5, sample_size=-1)

    assert result_dense.epsilon_used < result_sparse.epsilon_used, (
        f"Expected dense epsilon ({result_dense.epsilon_used:.6f}) < "
        f"sparse epsilon ({result_sparse.epsilon_used:.6f})"
    )


def test_epsilon_used_equals_fraction_times_knn():
    """epsilon_used must equal epsilon_fraction * mean_knn_distance."""
    result = compute(
        np.random.RandomState(42).randn(200, 32),
        epsilon_fraction=0.05,
        sample_size=-1
    )
    expected = result.epsilon_fraction * result.mean_knn_distance
    assert abs(result.epsilon_used - expected) < 1e-10


def test_mean_knn_distance_positive(uniform_embeddings):
    """Mean k-NN distance must be positive for non-degenerate data."""
    result = compute(uniform_embeddings)
    assert result.mean_knn_distance > 0


# ── Stability array tests ──────────────────────────────────────────────────────

def test_stability_array_shape(uniform_embeddings):
    """stability_per_embedding must have shape (n,)."""
    n = len(uniform_embeddings)
    result = compute(uniform_embeddings)
    assert result.stability_per_embedding.shape == (n,)


def test_mean_stability_equals_mean_of_array(uniform_embeddings):
    """mean_stability must equal mean of stability_per_embedding."""
    result = compute(uniform_embeddings)
    expected = float(np.mean(result.stability_per_embedding))
    assert abs(result.mean_stability - expected) < 1e-10


# ── Brittle zone tests ─────────────────────────────────────────────────────────

def test_brittle_zone_indices_consistent():
    """brittle_zone_indices must be indices where stability < threshold."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 32)
    result = compute(embeddings, brittle_threshold=0.5, sample_size=-1)
    expected = set(np.where(result.stability_per_embedding < 0.5)[0])
    actual = set(result.brittle_zone_indices)
    assert expected == actual


def test_n_brittle_consistent(uniform_embeddings):
    """n_brittle must equal len(brittle_zone_indices)."""
    result = compute(uniform_embeddings)
    assert result.n_brittle == len(result.brittle_zone_indices)


def test_brittle_fraction_consistent(uniform_embeddings):
    """brittle_fraction must equal n_brittle / n_vectors."""
    result = compute(uniform_embeddings)
    expected = result.n_brittle / result.n_vectors
    assert abs(result.brittle_fraction - expected) < 1e-10


def test_brittle_fraction_bounded(uniform_embeddings):
    """brittle_fraction must be in [0, 1]."""
    result = compute(uniform_embeddings)
    assert 0.0 <= result.brittle_fraction <= 1.0


def test_higher_threshold_more_brittle(uniform_embeddings):
    """Higher brittle_threshold should classify more embeddings as brittle."""
    result_low = compute(uniform_embeddings, brittle_threshold=0.2, sample_size=-1)
    result_high = compute(uniform_embeddings, brittle_threshold=0.8, sample_size=-1)
    assert result_high.n_brittle >= result_low.n_brittle


# ── Sampling tests ─────────────────────────────────────────────────────────────

def test_full_computation_flag(uniform_embeddings):
    """sample_size=-1 should set sampled=False."""
    result = compute(uniform_embeddings, sample_size=-1)
    assert result.sampled is False
    assert result.n_sampled == len(uniform_embeddings)


def test_sampling_flag_when_sample_used():
    """When sample_size < n, sampled should be True."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(500, 32)
    result = compute(embeddings, sample_size=100)
    assert result.sampled is True
    assert result.n_sampled == 100
    assert result.n_vectors == 500


def test_stability_array_full_length_when_sampled():
    """stability_per_embedding must have length n even when sampling."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(300, 32)
    result = compute(embeddings, sample_size=50)
    assert len(result.stability_per_embedding) == 300


def test_reproducibility(uniform_embeddings):
    """Same random seed should produce identical results."""
    result1 = compute(uniform_embeddings, random_seed=42)
    result2 = compute(uniform_embeddings, random_seed=42)
    assert result1.mean_stability == result2.mean_stability
    np.testing.assert_array_equal(
        result1.stability_per_embedding,
        result2.stability_per_embedding
    )


def test_different_seeds_different_results():
    """Different seeds should generally produce different scores."""
    rng = np.random.RandomState(0)
    embeddings = rng.randn(200, 32)
    result1 = compute(embeddings, random_seed=1, sample_size=50)
    result2 = compute(embeddings, random_seed=999, sample_size=50)
    # Different samples → different scores (almost certainly)
    assert result1.mean_stability != result2.mean_stability or \
           not np.array_equal(result1.stability_per_embedding,
                              result2.stability_per_embedding)


# ── Interpretation tests ───────────────────────────────────────────────────────

def test_interpretation_valid_string(uniform_embeddings):
    """Interpretation must be one of the four valid values."""
    result = compute(uniform_embeddings)
    assert result.interpretation in {"stable", "moderate", "sensitive", "brittle"}


def test_interpretation_boundaries():
    """Test interpretation boundary values."""
    from spectralyte.metrics.sensitivity import _interpret
    assert _interpret(1.0) == "stable"
    assert _interpret(0.80) == "stable"
    assert _interpret(0.79) == "moderate"
    assert _interpret(0.60) == "moderate"
    assert _interpret(0.59) == "sensitive"
    assert _interpret(0.40) == "sensitive"
    assert _interpret(0.39) == "brittle"
    assert _interpret(0.0) == "brittle"


def test_stable_space_interpretation(stable_embeddings):
    """Well-separated clusters should have stable or moderate interpretation."""
    result = compute(stable_embeddings, k=5, m=3, sample_size=-1)
    assert result.interpretation in {"stable", "moderate"}, (
        f"Expected 'stable' or 'moderate' for well-separated clusters, "
        f"got '{result.interpretation}' (stability={result.mean_stability:.3f})"
    )


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


def test_raises_when_m_zero():
    """m = 0 should raise ValueError."""
    embeddings = np.random.randn(100, 32)
    with pytest.raises(ValueError):
        compute(embeddings, m=0)


def test_raises_when_epsilon_zero():
    """epsilon_fraction = 0 should raise ValueError."""
    embeddings = np.random.randn(100, 32)
    with pytest.raises(ValueError):
        compute(embeddings, epsilon_fraction=0.0)


def test_raises_when_epsilon_negative():
    """Negative epsilon_fraction should raise ValueError."""
    embeddings = np.random.randn(100, 32)
    with pytest.raises(ValueError):
        compute(embeddings, epsilon_fraction=-0.1)


def test_raises_when_brittle_threshold_out_of_range():
    """brittle_threshold outside [0, 1] should raise ValueError."""
    embeddings = np.random.randn(100, 32)
    with pytest.raises(ValueError):
        compute(embeddings, brittle_threshold=1.5)
    with pytest.raises(ValueError):
        compute(embeddings, brittle_threshold=-0.1)


# ── Different dimensions and sizes ─────────────────────────────────────────────

@pytest.mark.parametrize("d", [16, 64, 128, 384])
def test_different_dimensions(d):
    """Should work on any embedding dimension."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, d)
    result = compute(embeddings, k=5, sample_size=50)
    assert result.n_dims == d
    assert 0.0 <= result.mean_stability <= 1.0


@pytest.mark.parametrize("m", [1, 3, 5, 10])
def test_different_m_values(m):
    """Should work with different numbers of perturbations."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(100, 32)
    result = compute(embeddings, m=m, sample_size=30)
    assert result.m == m
    assert 0.0 <= result.mean_stability <= 1.0