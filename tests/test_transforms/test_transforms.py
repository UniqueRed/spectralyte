"""
tests/test_transforms/test_transforms.py
=========================================
Mathematical property tests for the three correction transforms.

Shape, dispatch and error-handling coverage lives in
tests/test_core/test_auditor.py. This module verifies that each transform
actually does what its name claims:

- whiten     — output covariance is isotropic, rows are unit norm
- abtt       — the top-k principal directions are genuinely projected out
- pca_reduce — output columns are decorrelated and rank-limited

Together these are the properties downstream code relies on; a transform
that returned the right shape but the wrong geometry would pass the
auditor tests and still silently corrupt an index.
"""

import numpy as np
import pytest

from spectralyte import Spectralyte


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def anisotropic_embeddings():
    """
    Embeddings carrying BOTH geometric pathologies the transforms target.

    These are distinct problems and a fixture needs both, or half these
    tests go vacuous:

    - A skewed covariance spectrum (dimensions scaled on a geometric
      ladder, then mixed by a random rotation). This is what whitening
      flattens. A constant bias vector would not do it — centering
      removes a bias outright and leaves the covariance untouched.

    - A shared dominant direction (a large common offset). This is what
      the anisotropy metric actually measures, via mean pairwise cosine
      similarity, and what ABTT strips out. Skewed covariance alone
      leaves vectors spread symmetrically about the origin and scores a
      perfectly healthy 0.0.
    """
    rng = np.random.RandomState(42)

    base = rng.randn(200, 32)
    scales = np.logspace(0, -2, 32)          # 100x spread across dimensions
    mixing = rng.randn(32, 32)
    correlated = (base * scales) @ mixing

    direction = rng.randn(32)
    direction /= np.linalg.norm(direction)
    offset = direction * np.abs(correlated).mean() * 20.0

    return correlated + offset


@pytest.fixture
def audit(anisotropic_embeddings):
    """Spectralyte instance with a completed audit."""
    a = Spectralyte(anisotropic_embeddings, k=5, random_seed=42)
    a.run(verbose=False)
    return a


def _centered(x):
    return x - x.mean(axis=0)


def _top_right_singular_vectors(x, k):
    """Top-k right singular vectors of centered x, shape (d, k)."""
    _, _, Vt = np.linalg.svd(_centered(x), full_matrices=False)
    return Vt[:k, :].T


# ── whiten ─────────────────────────────────────────────────────────────────────

def test_whiten_rows_are_unit_norm(audit, anisotropic_embeddings):
    """whiten L2-normalizes its output, so every row has norm 1."""
    out = audit.transform(anisotropic_embeddings, strategy="whiten")
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_whiten_decorrelates_dimensions(audit, anisotropic_embeddings):
    """
    Whitening drives the covariance toward identity, so off-diagonal
    covariance must be far smaller than the diagonal.
    """
    out = audit.transform(anisotropic_embeddings, strategy="whiten")
    cov = np.cov(_centered(out), rowvar=False)

    diag = np.abs(np.diag(cov)).mean()
    off = np.abs(cov - np.diag(np.diag(cov))).mean()

    assert off < diag * 0.1


def test_whiten_flattens_eigenvalue_spectrum(audit, anisotropic_embeddings):
    """
    The point of whitening: the eigenvalue spread of the covariance
    collapses toward uniform. Compare condition-number-like ratios.
    """
    def spread(x):
        ev = np.linalg.eigvalsh(np.cov(_centered(x), rowvar=False))
        ev = np.clip(ev, 1e-12, None)
        return ev.max() / ev.min()

    before = spread(anisotropic_embeddings)
    after = spread(audit.transform(anisotropic_embeddings, strategy="whiten"))

    assert after < before


def test_whiten_handles_rank_deficient_input(audit):
    """
    A rank-deficient matrix produces near-zero eigenvalues. The
    implementation clips them at 1e-10, so the result must stay finite
    rather than blowing up to inf/nan.
    """
    rng = np.random.RandomState(0)
    latent = rng.randn(60, 3)
    basis = rng.randn(3, 32)
    degenerate = latent @ basis          # rank 3 in 32 nominal dims

    out = audit.transform(degenerate, strategy="whiten")
    assert np.all(np.isfinite(out))


# ── abtt ───────────────────────────────────────────────────────────────────────

def test_abtt_rows_are_unit_norm(audit, anisotropic_embeddings):
    """abtt L2-normalizes its output."""
    out = audit.transform(anisotropic_embeddings, strategy="abtt")
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_abtt_removes_top_k_directions(audit, anisotropic_embeddings):
    """
    The defining property: after ABTT the embeddings carry no component
    along the top-k principal directions of the original matrix.

    Row-wise L2 normalization is a per-row rescale, so it preserves the
    orthogonality established by the projection.
    """
    k = 3
    W_k = _top_right_singular_vectors(anisotropic_embeddings, k)
    out = audit.transform(anisotropic_embeddings, strategy="abtt", abtt_k=k)

    residual = out @ W_k              # (n, k) — should be ~0
    assert np.abs(residual).max() < 1e-8


def test_abtt_k_controls_how_many_directions_are_removed(audit, anisotropic_embeddings):
    """A larger abtt_k must project out strictly more directions."""
    out1 = audit.transform(anisotropic_embeddings, strategy="abtt", abtt_k=1)
    out5 = audit.transform(anisotropic_embeddings, strategy="abtt", abtt_k=5)

    W_5 = _top_right_singular_vectors(anisotropic_embeddings, 5)

    # abtt_k=5 zeroes all five directions; abtt_k=1 leaves directions 2-5 intact.
    assert np.abs(out5 @ W_5).max() < 1e-8
    assert np.abs(out1 @ W_5).max() > 1e-3


def test_abtt_reduces_anisotropy(anisotropic_embeddings):
    """Removing the dominant directions should lower the anisotropy score."""
    audit = Spectralyte(anisotropic_embeddings, k=5, random_seed=42)
    before = audit.run(verbose=False)

    fixed = audit.transform(anisotropic_embeddings, strategy="abtt")
    after = Spectralyte(fixed, k=5, random_seed=42).run(verbose=False)

    assert after.anisotropy.score < before.anisotropy.score


# ── pca_reduce ─────────────────────────────────────────────────────────────────

def test_pca_reduce_matches_reported_effective_dims(audit, anisotropic_embeddings):
    """Output width must equal the effective_dims the audit measured."""
    report = audit.run(anisotropic_embeddings, verbose=False)
    out = audit.transform(anisotropic_embeddings, strategy="pca_reduce")

    assert out.shape == (anisotropic_embeddings.shape[0],
                         report.dimensionality.effective_dims)


def test_pca_reduce_decorrelates_components(audit, anisotropic_embeddings):
    """
    Principal components are mutually orthogonal, so the projected
    output must have near-diagonal covariance.
    """
    out = audit.transform(anisotropic_embeddings, strategy="pca_reduce")
    if out.shape[1] < 2:
        pytest.skip("need at least 2 components to test decorrelation")

    cov = np.cov(out, rowvar=False)
    off = np.abs(cov - np.diag(np.diag(cov))).max()
    diag = np.abs(np.diag(cov)).max()

    assert off < diag * 1e-6


def test_pca_reduce_preserves_dominant_variance(audit, anisotropic_embeddings):
    """
    Reduction drops noise dimensions, not signal: the retained variance
    must be the bulk of the original.
    """
    out = audit.transform(anisotropic_embeddings, strategy="pca_reduce")

    total = _centered(anisotropic_embeddings).var(axis=0).sum()
    kept = out.var(axis=0).sum()

    assert kept / total > 0.9


# ── Cross-cutting ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy", ["whiten", "abtt", "pca_reduce"])
def test_transform_does_not_mutate_input(audit, anisotropic_embeddings, strategy):
    """Transforms must return a new array, never edit the caller's matrix."""
    original = anisotropic_embeddings.copy()
    audit.transform(anisotropic_embeddings, strategy=strategy)
    assert np.array_equal(anisotropic_embeddings, original)


@pytest.mark.parametrize("strategy", ["whiten", "abtt", "pca_reduce"])
def test_transform_output_is_finite(audit, anisotropic_embeddings, strategy):
    """No transform may emit NaN or inf."""
    out = audit.transform(anisotropic_embeddings, strategy=strategy)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("strategy", ["whiten", "abtt", "pca_reduce"])
def test_transform_is_deterministic(audit, anisotropic_embeddings, strategy):
    """Repeated calls on the same input must give identical results."""
    a = audit.transform(anisotropic_embeddings, strategy=strategy)
    b = audit.transform(anisotropic_embeddings, strategy=strategy)
    assert np.array_equal(a, b)
