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
import re
import tempfile
import os
from spectralyte import Spectralyte
from spectralyte.core import severity
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

# ── fix_plan() / n_issues coherence ────────────────────────────────────────────
#
# These lock the invariant that fix_plan() addresses exactly the metrics
# n_issues counts. The two used to disagree: n_issues counted any non-healthy
# metric, while fix_plan() carried a hand-maintained if-chain that skipped the
# "moderate" tier for three of the five metrics and had no branch at all for
# intrinsic_dim. A report could print "3 issues detected — run fix_plan()" and
# then hand back a plan addressing one of them.

def _plan_issue_numbers(plan):
    return re.findall(r"^Issue (\d+):", plan, re.M)


@pytest.fixture
def collapsed_embeddings():
    """Rank-3 data in 96 nominal dims — trips intrinsic_dim and dimensionality."""
    rng = np.random.default_rng(11)
    return (rng.normal(size=(300, 3)) @ rng.normal(size=(3, 96))).astype(np.float32)


@pytest.fixture
def clustered_embeddings():
    """Two well-separated clusters — trips the density metric."""
    rng = np.random.default_rng(5)
    return np.vstack([rng.normal(size=(150, 32)),
                      rng.normal(size=(150, 32)) + 18]).astype(np.float32)


@pytest.mark.parametrize(
    "fixture_name",
    ["small_embeddings", "collapsed_embeddings", "clustered_embeddings"],
)
def test_fix_plan_section_count_matches_n_issues(fixture_name, request):
    """Every metric n_issues counts must get a section, and no others."""
    emb = request.getfixturevalue(fixture_name)
    report = Spectralyte(emb, k=5, random_seed=42).run(verbose=False)

    assert len(_plan_issue_numbers(report.fix_plan())) == report.n_issues


@pytest.mark.parametrize(
    "fixture_name",
    ["small_embeddings", "collapsed_embeddings", "clustered_embeddings"],
)
def test_fix_plan_numbering_is_sequential(fixture_name, request):
    """Issues are numbered 1..n in emission order, never skipping."""
    emb = request.getfixturevalue(fixture_name)
    report = Spectralyte(emb, k=5, random_seed=42).run(verbose=False)

    nums = _plan_issue_numbers(report.fix_plan())
    assert nums == [str(i + 1) for i in range(len(nums))]


def test_fix_plan_covers_collapsed_manifold(collapsed_embeddings):
    """
    A collapsed manifold is the most serious finding Spectralyte makes and
    must produce guidance, not silence. It previously had no branch at all.
    """
    report = Spectralyte(collapsed_embeddings, k=5, random_seed=42).run(verbose=False)
    # Graded WARN rather than BAD since the benchmark found the label carries
    # no retrieval-predictive power, but it must still produce guidance.
    assert severity.severity(report.intrinsic_dim) == severity.WARN

    plan = report.fix_plan()
    assert "Collapsed Manifold" in plan
    # The honest part: transforms cannot undo a collapse.
    assert "no transform fixes this" in plan


def test_fix_plan_distinguishes_warn_from_bad(collapsed_embeddings):
    """Sections are tagged by severity so a borderline reading is not
    presented with the same urgency as an active failure."""
    report = Spectralyte(collapsed_embeddings, k=5, random_seed=42).run(verbose=False)
    plan = report.fix_plan()

    levels = [severity.severity(getattr(report, m)) for m in severity.METRIC_NAMES]
    if severity.BAD in levels:
        assert "[CRITICAL]" in plan
    if severity.WARN in levels:
        assert "[WARNING]" in plan
        assert "Borderline reading" in plan


def test_fix_plan_healthy_report_has_no_sections():
    """A clean space yields no numbered issues at all."""
    rng = np.random.default_rng(3)
    emb = rng.normal(size=(300, 64)).astype(np.float32)
    report = Spectralyte(emb, k=5, random_seed=42).run(verbose=False)

    assert report.n_issues == 0
    plan = report.fix_plan()
    assert _plan_issue_numbers(plan) == []
    assert "No issues detected" in plan


def test_needs_transform_agrees_with_severity(collapsed_embeddings):
    """needs_transform is graded through severity, not its own thresholds."""
    report = Spectralyte(collapsed_embeddings, k=5, random_seed=42).run(verbose=False)

    expected = (not severity.is_healthy(report.anisotropy)
                or not severity.is_healthy(report.dimensionality))
    assert report.needs_transform == expected


# ── Conditioning-aware remediation ─────────────────────────────────────────────

def _anisotropic(rng, cond="well"):
    d = rng.randn(32)
    d /= np.linalg.norm(d)
    if cond == "well":
        return rng.randn(300, 32) + d * 6.0
    base = (rng.randn(300, 32) * np.logspace(0, -2, 32)) @ rng.randn(32, 32)
    return base + d * np.abs(base).mean() * 20.0


def test_condition_number_tracks_the_spectrum():
    """condition_number must separate an even spectrum from a degenerate one."""
    rng = np.random.RandomState(42)
    well = Spectralyte(_anisotropic(rng, "well"), k=5, random_seed=42).run(verbose=False)
    ill = Spectralyte(_anisotropic(rng, "ill"), k=5, random_seed=42).run(verbose=False)

    assert well.condition_number < 1e3
    assert ill.condition_number > 1e4


def test_fix_plan_recommends_whitening_when_well_conditioned():
    rng = np.random.RandomState(42)
    report = Spectralyte(_anisotropic(rng, "well"), k=5, random_seed=42).run(verbose=False)

    assert severity.severity(report.anisotropy) != severity.OK
    plan = report.fix_plan()
    assert "strategy='whiten'" in plan
    assert "Recommending ABTT" not in plan


def test_fix_plan_recommends_whitening_regardless_of_conditioning():
    """
    An earlier heuristic steered ill-conditioned spaces to ABTT. Benchmarking
    killed it: real embedding spaces all have enormous condition numbers
    (6.9e34 to 3.8e38 for healthy sentence encoders), so the threshold fired
    on every space and discriminated nothing — and whitening beat ABTT on both
    pathological cases anyway (nDCG@10 0.319 vs 0.274 on SciFact, 0.063 vs
    0.059 on NFCorpus).
    """
    rng = np.random.RandomState(42)
    report = Spectralyte(_anisotropic(rng, "ill"), k=5, random_seed=42).run(verbose=False)

    assert report.condition_number > 1e4        # would have tripped the old steer
    assert severity.severity(report.anisotropy) != severity.OK

    plan = report.fix_plan()
    assert "strategy='whiten'" in plan
    assert "Recommending ABTT" not in plan
