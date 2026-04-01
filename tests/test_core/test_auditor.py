"""
tests/test_core/test_auditor.py
================================
Integration tests for the Spectralyte orchestrator and AuditReport.

Tests the full audit pipeline end-to-end, verifying that:
- Spectralyte.run() produces a complete AuditReport
- All five metrics are computed and accessible
- AuditReport properties are correct
- transform() applies correctly
- fix_plan() generates output for all frameworks
- export() produces valid JSON
- Input validation raises appropriate errors
"""

import json
import numpy as np
import pytest
import tempfile
import os
from spectralyte import Spectralyte
from spectralyte.core.report import AuditReport


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_embeddings():
    """Small embedding matrix for fast tests."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 64)


@pytest.fixture
def audit(small_embeddings):
    """Spectralyte instance with small embeddings."""
    return Spectralyte(small_embeddings, k=5, random_seed=42)


@pytest.fixture
def report(audit):
    """Pre-computed audit report."""
    return audit.run(verbose=False)


# ── Spectralyte initialization tests ──────────────────────────────────────────

def test_init_stores_embeddings(small_embeddings):
    """Spectralyte must store the embedding matrix."""
    audit = Spectralyte(small_embeddings)
    assert audit.embeddings is small_embeddings


def test_init_stores_config():
    """Configuration parameters must be stored."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(50, 32)
    audit = Spectralyte(
        embeddings, k=7, sensitivity_epsilon=0.1,
        sensitivity_m=3, variance_threshold=0.90,
        random_seed=99
    )
    assert audit.k == 7
    assert audit.sensitivity_epsilon == 0.1
    assert audit.sensitivity_m == 3
    assert audit.variance_threshold == 0.90
    assert audit.random_seed == 99


def test_init_raises_for_1d():
    """1D input should raise ValueError."""
    with pytest.raises(ValueError, match="2D"):
        Spectralyte(np.array([1.0, 2.0, 3.0]))


def test_init_raises_for_too_few_vectors():
    """Fewer than 3 embeddings should raise ValueError."""
    with pytest.raises(ValueError, match="at least 3"):
        Spectralyte(np.array([[1.0, 2.0], [3.0, 4.0]]))


# ── run() tests ───────────────────────────────────────────────────────────────

def test_run_returns_audit_report(report):
    """run() must return an AuditReport."""
    assert isinstance(report, AuditReport)


def test_run_has_all_five_metrics(report):
    """AuditReport must have all five metric results."""
    assert report.anisotropy is not None
    assert report.dimensionality is not None
    assert report.density is not None
    assert report.sensitivity is not None
    assert report.intrinsic_dim is not None


def test_run_embeddings_shape_correct(report, small_embeddings):
    """embeddings_shape must match the input matrix."""
    assert report.embeddings_shape == small_embeddings.shape


def test_run_anisotropy_score_bounded(report):
    """Anisotropy score must be in [0, 1]."""
    assert 0.0 <= report.anisotropy.score <= 1.0


def test_run_effective_dims_bounded(report, small_embeddings):
    """Effective dims must be >= 1 and <= nominal dims."""
    d = small_embeddings.shape[1]
    assert 1 <= report.dimensionality.effective_dims <= d


def test_run_density_cv_non_negative(report):
    """Density CV must be non-negative."""
    assert report.density.cv >= 0.0


def test_run_stability_bounded(report):
    """Mean stability must be in [0, 1]."""
    assert 0.0 <= report.sensitivity.mean_stability <= 1.0


def test_run_intrinsic_dim_positive(report):
    """Intrinsic dimensionality must be positive."""
    assert report.intrinsic_dim.d_int >= 1.0


def test_run_with_alternate_embeddings(audit):
    """run() should accept a different embedding matrix."""
    rng = np.random.RandomState(99)
    other = rng.randn(80, 64)
    report = audit.run(other, verbose=False)
    assert report.embeddings_shape == (80, 64)


def test_run_reproducible(small_embeddings):
    """Same seed should produce identical results."""
    audit1 = Spectralyte(small_embeddings, random_seed=42)
    audit2 = Spectralyte(small_embeddings, random_seed=42)
    r1 = audit1.run(verbose=False)
    r2 = audit2.run(verbose=False)
    assert r1.anisotropy.score == r2.anisotropy.score
    assert r1.sensitivity.mean_stability == r2.sensitivity.mean_stability


# ── AuditReport property tests ─────────────────────────────────────────────────

def test_n_issues_is_non_negative(report):
    """n_issues must be >= 0."""
    assert report.n_issues >= 0


def test_n_issues_bounded(report):
    """n_issues must be <= 5 (one per metric)."""
    assert report.n_issues <= 5


def test_needs_transform_is_bool(report):
    """needs_transform must be a boolean."""
    assert isinstance(report.needs_transform, bool)


def test_has_brittle_zones_is_bool(report):
    """has_brittle_zones must be a boolean."""
    assert isinstance(report.has_brittle_zones, bool)


def test_has_brittle_zones_consistent(report):
    """has_brittle_zones must be consistent with n_brittle."""
    if report.sensitivity.n_brittle > 0:
        assert report.has_brittle_zones is True
    else:
        assert report.has_brittle_zones is False


# ── summary() tests ────────────────────────────────────────────────────────────

def test_summary_runs_without_error(report, capsys):
    """summary() must complete without raising."""
    report.summary(use_color=False)
    captured = capsys.readouterr()
    assert "Spectralyte Audit Report" in captured.out


def test_summary_contains_all_metrics(report, capsys):
    """summary() output must mention all five metrics."""
    report.summary(use_color=False)
    captured = capsys.readouterr()
    assert "Anisotropy" in captured.out
    assert "Effective Dimensions" in captured.out
    assert "Density" in captured.out
    assert "Retrieval Stability" in captured.out
    assert "Intrinsic" in captured.out


def test_summary_no_color_is_plain_text(report, capsys):
    """use_color=False should produce no ANSI escape codes."""
    report.summary(use_color=False)
    captured = capsys.readouterr()
    assert "\033[" not in captured.out


# ── fix_plan() tests ───────────────────────────────────────────────────────────

def test_fix_plan_returns_string(report):
    """fix_plan() must return a string."""
    plan = report.fix_plan()
    assert isinstance(plan, str)


def test_fix_plan_non_empty(report):
    """fix_plan() must return non-empty string."""
    plan = report.fix_plan()
    assert len(plan) > 0


def test_fix_plan_contains_header(report):
    """fix_plan() output must contain the Spectralyte header."""
    plan = report.fix_plan()
    assert "Spectralyte" in plan
    assert "Remediation Plan" in plan


@pytest.mark.parametrize("framework", ["generic", "langchain", "llamaindex"])
def test_fix_plan_all_frameworks(report, framework):
    """fix_plan() should work for all supported frameworks."""
    plan = report.fix_plan(framework=framework)
    assert isinstance(plan, str)
    assert len(plan) > 0


# ── export() tests ─────────────────────────────────────────────────────────────

def test_export_creates_file(report):
    """export() must create a JSON file at the specified path."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        report.export(path)
        assert os.path.exists(path)
    finally:
        os.unlink(path)


def test_export_produces_valid_json(report):
    """export() output must be valid JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        path = f.name
    try:
        report.export(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
    finally:
        os.unlink(path)


def test_export_contains_all_metrics(report):
    """Exported JSON must contain all five metric sections."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        path = f.name
    try:
        report.export(path)
        with open(path) as f:
            data = json.load(f)
        assert "anisotropy" in data
        assert "dimensionality" in data
        assert "density" in data
        assert "sensitivity" in data
        assert "intrinsic_dim" in data
    finally:
        os.unlink(path)


def test_export_scores_match_report(report):
    """Exported scores must match the in-memory report."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
        path = f.name
    try:
        report.export(path)
        with open(path) as f:
            data = json.load(f)
        assert abs(data["anisotropy"]["score"] - report.anisotropy.score) < 1e-10
        assert data["dimensionality"]["effective_dims"] == report.dimensionality.effective_dims
        assert abs(data["density"]["cv"] - report.density.cv) < 1e-10
    finally:
        os.unlink(path)


# ── transform() tests ──────────────────────────────────────────────────────────

def test_transform_requires_run_first(small_embeddings):
    """transform() before run() should raise RuntimeError."""
    audit = Spectralyte(small_embeddings)
    with pytest.raises(RuntimeError, match="run()"):
        audit.transform(small_embeddings)


def test_transform_whiten_same_shape(audit, report, small_embeddings):
    """Whitening must return same shape as input."""
    fixed = audit.transform(small_embeddings, strategy="whiten")
    assert fixed.shape == small_embeddings.shape


def test_transform_abtt_same_shape(audit, report, small_embeddings):
    """ABTT must return same shape as input."""
    fixed = audit.transform(small_embeddings, strategy="abtt")
    assert fixed.shape == small_embeddings.shape


def test_transform_pca_reduce_correct_shape(audit, report, small_embeddings):
    """PCA reduce must return shape (n, effective_dims)."""
    fixed = audit.transform(small_embeddings, strategy="pca_reduce")
    expected_dims = report.dimensionality.effective_dims
    assert fixed.shape == (small_embeddings.shape[0], expected_dims)


def test_transform_whiten_reduces_anisotropy(small_embeddings):
    """Whitening should reduce anisotropy score."""
    rng = np.random.RandomState(42)
    # Create anisotropic embeddings
    base = rng.randn(64)
    base = base / np.linalg.norm(base)
    anisotropic = base + rng.randn(200, 64) * 0.1

    audit = Spectralyte(anisotropic, k=5, random_seed=42)
    report_before = audit.run(verbose=False)
    fixed = audit.transform(anisotropic, strategy="whiten")

    audit2 = Spectralyte(fixed, k=5, random_seed=42)
    report_after = audit2.run(verbose=False)

    assert report_after.anisotropy.score < report_before.anisotropy.score, (
        f"Whitening should reduce anisotropy: "
        f"before={report_before.anisotropy.score:.4f}, "
        f"after={report_after.anisotropy.score:.4f}"
    )


def test_transform_invalid_strategy(audit, report, small_embeddings):
    """Invalid strategy should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        audit.transform(small_embeddings, strategy="invalid")


# ── get_router() tests ─────────────────────────────────────────────────────────

def test_get_router_requires_run_first(small_embeddings):
    """get_router() before run() should raise RuntimeError."""
    audit = Spectralyte(small_embeddings)
    with pytest.raises(RuntimeError, match="run()"):
        audit.get_router()


def test_get_router_returns_router(audit, report):
    """get_router() must return a Router instance."""
    from spectralyte.core.router import Router
    router = audit.get_router()
    assert isinstance(router, Router)


# ── compare() tests ────────────────────────────────────────────────────────────

def test_compare_without_transform_prints_message(report, capsys):
    """compare() without a pre-transform report should print a message."""
    report.compare()
    captured = capsys.readouterr()
    assert "No pre-transform report" in captured.out


def test_compare_after_transform_shows_table(small_embeddings, capsys):
    """compare() after transform should show before/after table."""
    audit = Spectralyte(small_embeddings, k=5, random_seed=42)
    audit.run(verbose=False)
    fixed = audit.transform(small_embeddings, strategy="whiten")
    report_after = audit.run(fixed, verbose=False)
    report_after.compare(use_color=False)
    captured = capsys.readouterr()
    assert "Before" in captured.out
    assert "After" in captured.out