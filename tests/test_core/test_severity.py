"""
Tests for spectralyte.core.severity — the canonical per-metric grading of
interpretation labels.

These cover the regression that motivated the module: the five metrics do not
share a polarity, and their interpretation labels collide, so a single global
"healthy" label set silently mis-graded two of them.
"""

import pytest

from spectralyte.core import severity
from spectralyte.core.severity import OK, WARN, BAD, severity_of


# ── Per-metric grading ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("metric,label,expected", [
    # anisotropy: low is healthy
    ("anisotropy", "healthy", OK),
    ("anisotropy", "moderate", WARN),
    ("anisotropy", "high", BAD),
    ("anisotropy", "severe", BAD),
    # dimensionality: high utilization is healthy
    ("dimensionality", "healthy", OK),
    ("dimensionality", "moderate", WARN),
    ("dimensionality", "low", BAD),
    ("dimensionality", "critical", BAD),
    # density: uniform is healthy
    ("density", "uniform", OK),
    ("density", "moderate", WARN),
    ("density", "clustered", BAD),
    ("density", "severe", BAD),
    # sensitivity: stable is healthy
    ("sensitivity", "stable", OK),
    ("sensitivity", "moderate", WARN),
    ("sensitivity", "sensitive", BAD),
    ("sensitivity", "brittle", BAD),
    # intrinsic_dim: INVERTED — only a collapsed manifold is a problem
    ("intrinsic_dim", "very_high", OK),
    ("intrinsic_dim", "high", OK),
    ("intrinsic_dim", "moderate", OK),
    ("intrinsic_dim", "low", BAD),
])
def test_severity_of(metric, label, expected):
    assert severity_of(metric, label) == expected


# ── The collisions that made a global label set unworkable ─────────────────────

def test_low_is_bad_for_both_metrics_that_emit_it():
    """
    Regression: "low" was in a shared healthy set.

    Both metrics that emit it mean something unhealthy — an under-utilized
    space (dimensionality) and a collapsed manifold (intrinsic_dim).
    """
    assert severity_of("dimensionality", "low") == BAD
    assert severity_of("intrinsic_dim", "low") == BAD


def test_severe_is_bad_for_both_metrics_that_emit_it():
    assert severity_of("anisotropy", "severe") == BAD
    assert severity_of("density", "severe") == BAD


def test_intrinsic_dim_polarity_is_inverted_vs_other_metrics():
    """A high intrinsic-dim ratio is healthy; for other metrics it is not."""
    assert severity_of("intrinsic_dim", "high") == OK
    assert severity_of("anisotropy", "high") == BAD


def test_dimensionality_grading_is_monotonic():
    """
    Regression: "low" ranked healthier than the strictly better "moderate".

    Severity must not improve as utilization falls.
    """
    order = {OK: 0, WARN: 1, BAD: 2}
    ranked = [severity_of("dimensionality", label)
              for label in ("healthy", "moderate", "low", "critical")]
    assert [order[r] for r in ranked] == sorted(order[r] for r in ranked)


# ── Fail-safe behavior ─────────────────────────────────────────────────────────

def test_unknown_metric_degrades_to_bad():
    """An unmapped metric must surface, not silently report healthy."""
    assert severity_of("not_a_metric", "healthy") == BAD


def test_unknown_label_degrades_to_bad():
    assert severity_of("anisotropy", "not_a_label") == BAD


# ── Result-object helpers ──────────────────────────────────────────────────────

class _Res:
    def __init__(self, metric, interpretation):
        self.METRIC = metric
        self.interpretation = interpretation


def test_severity_reads_metric_from_result():
    assert severity.severity(_Res("intrinsic_dim", "high")) == OK
    assert severity.severity(_Res("anisotropy", "high")) == BAD


def test_is_healthy():
    assert severity.is_healthy(_Res("density", "uniform"))
    assert not severity.is_healthy(_Res("density", "clustered"))


def test_result_without_metric_attribute_degrades_to_bad():
    class Bare:
        interpretation = "healthy"

    assert severity.severity(Bare()) == BAD


# ── Metric registry stays in sync with AuditReport ─────────────────────────────

def test_metric_names_match_report_attributes():
    """Every name in METRIC_NAMES must be a real AuditReport attribute."""
    import dataclasses

    from spectralyte.core.report import AuditReport

    fields = {f.name for f in dataclasses.fields(AuditReport)}
    for name in severity.METRIC_NAMES:
        assert name in fields, f"{name} is not an AuditReport field"


def test_every_result_dataclass_declares_its_metric():
    """Each metric result must carry a METRIC matching the severity table."""
    from spectralyte.metrics.anisotropy import AnisotropyResult
    from spectralyte.metrics.density import DensityResult
    from spectralyte.metrics.dimensionality import DimensionalityResult
    from spectralyte.metrics.intrinsic_dim import IntrinsicDimResult
    from spectralyte.metrics.sensitivity import SensitivityResult

    declared = {
        AnisotropyResult.METRIC,
        DimensionalityResult.METRIC,
        DensityResult.METRIC,
        SensitivityResult.METRIC,
        IntrinsicDimResult.METRIC,
    }
    assert declared == set(severity.METRIC_NAMES)


def test_metric_is_not_a_dataclass_field():
    """METRIC is class-level identity, not per-instance data."""
    import dataclasses

    from spectralyte.metrics.anisotropy import AnisotropyResult

    assert "METRIC" not in {f.name for f in dataclasses.fields(AnisotropyResult)}
