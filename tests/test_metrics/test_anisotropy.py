"""
Tests for anisotropy metric.

Key properties to verify:
- Near-zero score on uniformly random unit vectors (isotropic)
- Near-one score on near-identical vectors (fully anisotropic)
- Score is bounded [0, 1]
- Works on different dimensions and sample sizes
"""

import numpy as np
import pytest
from spectralyte.metrics.anisotropy import compute


def test_isotropic_space():
    """Random unit vectors should have near-zero anisotropy."""
    rng = np.random.RandomState(42)
    vectors = rng.randn(500, 384)
    # L2 normalize
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    result = compute(vectors)
    assert result.score < 0.15, f"Expected near-zero anisotropy, got {result.score}"


def test_anisotropic_space():
    """Near-identical vectors should have high anisotropy."""
    rng = np.random.RandomState(42)
    base = rng.randn(384)
    base = base / np.linalg.norm(base)
    # Small perturbations of the same direction
    vectors = base + rng.randn(500, 384) * 0.01
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    result = compute(vectors)
    assert result.score > 0.85, f"Expected high anisotropy, got {result.score}"


def test_score_is_bounded():
    """Anisotropy score must be in [0, 1]."""
    rng = np.random.RandomState(42)
    vectors = rng.randn(200, 128)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    result = compute(vectors)
    assert 0.0 <= result.score <= 1.0


def test_different_dimensions():
    """Should work on any embedding dimension."""
    rng = np.random.RandomState(42)
    for d in [64, 256, 768, 1536]:
        vectors = rng.randn(100, d)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        result = compute(vectors)
        assert result.score is not None