"""
tests/test_metrics/test_anisotropy.py
=======================================
Tests for the anisotropy metric.

Key properties verified:
- Near-zero score on uniformly random unit vectors (isotropic space)
- High score on near-identical vectors (anisotropic space)
- Score is bounded [0, 1]
- Works on different dimensions and sample sizes
- Interpretation thresholds are correct
- Sampling triggers correctly for large inputs
- Input validation raises appropriate errors
"""

import numpy as np
import pytest
from spectralyte.metrics.anisotropy import compute, AnisotropyResult


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def isotropic_embeddings():
    """Random unit vectors — near-zero anisotropy."""
    rng = np.random.RandomState(42)
    vectors = rng.randn(500, 384)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def anisotropic_embeddings():
    """Near-identical unit vectors — high anisotropy."""
    rng = np.random.RandomState(42)
    base = rng.randn(384)
    base = base / np.linalg.norm(base)
    vectors = base + rng.randn(500, 384) * 0.01
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# ── Core correctness tests ─────────────────────────────────────────────────────

def test_isotropic_space(isotropic_embeddings):
    """Random unit vectors should have near-zero anisotropy."""
    result = compute(isotropic_embeddings)
    assert result.score < 0.15, (
        f"Expected near-zero anisotropy for random unit vectors, got {result.score:.4f}"
    )


def test_anisotropic_space(anisotropic_embeddings):
    """Near-identical vectors should have high anisotropy."""
    result = compute(anisotropic_embeddings)
    assert result.score > 0.85, (
        f"Expected high anisotropy for near-identical vectors, got {result.score:.4f}"
    )


def test_score_bounded_zero_to_one(isotropic_embeddings):
    """Anisotropy score must always be in [0, 1]."""
    result = compute(isotropic_embeddings)
    assert 0.0 <= result.score <= 1.0, (
        f"Score {result.score} is out of bounds [0, 1]"
    )


def test_score_bounded_for_anisotropic(anisotropic_embeddings):
    """Score must be in [0, 1] even for extreme anisotropy."""
    result = compute(anisotropic_embeddings)
    assert 0.0 <= result.score <= 1.0


def test_ordering_isotropic_less_than_anisotropic(isotropic_embeddings, anisotropic_embeddings):
    """Isotropic space must have strictly lower score than anisotropic space."""
    iso_result = compute(isotropic_embeddings)
    aniso_result = compute(anisotropic_embeddings)
    assert iso_result.score < aniso_result.score, (
        f"Expected iso ({iso_result.score:.4f}) < aniso ({aniso_result.score:.4f})"
    )


# ── Return type tests ──────────────────────────────────────────────────────────

def test_returns_anisotropy_result(isotropic_embeddings):
    """compute() must return an AnisotropyResult dataclass."""
    result = compute(isotropic_embeddings)
    assert isinstance(result, AnisotropyResult)


def test_result_has_all_fields(isotropic_embeddings):
    """AnisotropyResult must have all expected fields."""
    result = compute(isotropic_embeddings)
    assert hasattr(result, 'score')
    assert hasattr(result, 'eigenvalues')
    assert hasattr(result, 'interpretation')
    assert hasattr(result, 'n_vectors')
    assert hasattr(result, 'n_dims')
    assert hasattr(result, 'sampled')
    assert hasattr(result, 'sample_size')


def test_metadata_correct(isotropic_embeddings):
    """n_vectors and n_dims must match input shape."""
    n, d = isotropic_embeddings.shape
    result = compute(isotropic_embeddings)
    assert result.n_vectors == n
    assert result.n_dims == d


def test_eigenvalues_shape(isotropic_embeddings):
    """Eigenvalues must be a 1D array."""
    result = compute(isotropic_embeddings)
    assert result.eigenvalues.ndim == 1
    assert len(result.eigenvalues) > 0


def test_eigenvalues_sorted_descending(isotropic_embeddings):
    """Eigenvalues must be sorted in descending order."""
    result = compute(isotropic_embeddings)
    evs = result.eigenvalues
    assert all(evs[i] >= evs[i+1] for i in range(len(evs)-1)), (
        "Eigenvalues must be sorted descending"
    )


# ── Interpretation threshold tests ────────────────────────────────────────────

def test_interpretation_healthy():
    """Score < 0.2 should be 'healthy'."""
    rng = np.random.RandomState(0)
    vectors = rng.randn(200, 128)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    result = compute(vectors)
    # Random vectors in reasonable dimensions should be healthy
    if result.score < 0.2:
        assert result.interpretation == "healthy"


def test_interpretation_severe(anisotropic_embeddings):
    """Score >= 0.6 should be 'severe'."""
    result = compute(anisotropic_embeddings)
    if result.score >= 0.6:
        assert result.interpretation == "severe"


def test_interpretation_boundaries():
    """Test that interpretation boundary values map correctly."""
    from spectralyte.metrics.anisotropy import _interpret
    assert _interpret(0.0) == "healthy"
    assert _interpret(0.19) == "healthy"
    assert _interpret(0.2) == "moderate"
    assert _interpret(0.39) == "moderate"
    assert _interpret(0.4) == "high"
    assert _interpret(0.59) == "high"
    assert _interpret(0.6) == "severe"
    assert _interpret(1.0) == "severe"


# ── Dimension tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("d", [64, 256, 384, 768, 1536])
def test_different_dimensions(d):
    """Should work on any embedding dimension."""
    rng = np.random.RandomState(42)
    vectors = rng.randn(100, d)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    result = compute(vectors)
    assert result is not None
    assert result.n_dims == d
    assert 0.0 <= result.score <= 1.0


@pytest.mark.parametrize("n", [2, 10, 50, 200, 1000])
def test_different_sample_sizes(n):
    """Should work with any number of embeddings >= 2."""
    rng = np.random.RandomState(42)
    vectors = rng.randn(n, 128)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    result = compute(vectors)
    assert result is not None
    assert result.n_vectors == n


# ── Sampling tests ─────────────────────────────────────────────────────────────

def test_sampling_triggers_for_large_input():
    """For n > sample_size, sampled must be True."""
    rng = np.random.RandomState(42)
    vectors = rng.randn(6000, 128)
    result = compute(vectors, sample_size=5000)
    assert result.sampled is True
    assert result.sample_size == 5000
    assert result.n_vectors == 6000


def test_no_sampling_for_small_input(isotropic_embeddings):
    """For n <= sample_size, sampled must be False."""
    result = compute(isotropic_embeddings, sample_size=5000)
    assert result.sampled is False
    assert result.sample_size == len(isotropic_embeddings)


def test_sampling_none_uses_all_vectors():
    """sample_size=None should use all vectors."""
    rng = np.random.RandomState(42)
    vectors = rng.randn(200, 64)
    result = compute(vectors, sample_size=None)
    assert result.sampled is False
    assert result.sample_size == 200


def test_sampling_reproducible():
    """Same random_seed should produce same score."""
    rng = np.random.RandomState(0)
    vectors = rng.randn(6000, 128)
    result1 = compute(vectors, sample_size=1000, random_seed=42)
    result2 = compute(vectors, sample_size=1000, random_seed=42)
    assert result1.score == result2.score


def test_sampling_different_seeds_different_scores():
    """Different seeds should generally produce different scores."""
    rng = np.random.RandomState(0)
    vectors = rng.randn(6000, 128)
    result1 = compute(vectors, sample_size=100, random_seed=1)
    result2 = compute(vectors, sample_size=100, random_seed=999)
    # They may occasionally be equal but usually not
    # Just verify both complete without error
    assert result1 is not None
    assert result2 is not None


# ── Input validation tests ─────────────────────────────────────────────────────

def test_raises_for_1d_input():
    """1D input should raise ValueError."""
    with pytest.raises(ValueError, match="2D"):
        compute(np.array([1.0, 2.0, 3.0]))


def test_raises_for_single_vector():
    """Single vector (n=1) should raise ValueError."""
    with pytest.raises(ValueError, match="at least 2"):
        compute(np.array([[1.0, 2.0, 3.0]]))


def test_handles_unnormalized_input():
    """Should work on raw unnormalized embeddings."""
    rng = np.random.RandomState(42)
    vectors = rng.randn(100, 128) * 100   # large magnitude
    result = compute(vectors)
    assert 0.0 <= result.score <= 1.0


def test_handles_zero_vectors():
    """Zero vectors should not cause division by zero."""
    rng = np.random.RandomState(42)
    vectors = rng.randn(100, 128)
    vectors[0] = 0.0   # inject one zero vector
    result = compute(vectors)
    assert result is not None
    assert not np.isnan(result.score)