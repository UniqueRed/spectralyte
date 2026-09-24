"""
spectralyte/core/severity.py
============================
Canonical mapping from a metric's interpretation label to a severity level.

Why this module exists
----------------------
The five metrics do NOT share a polarity. For four of them the healthy end is
the low end (low anisotropy, low density CV, high stability, high dimensionality
utilization). For intrinsic dimensionality the polarity is inverted: a *low*
``d_int / nominal_dims`` ratio means the manifold has collapsed onto a handful
of directions, which is the pathological case, while a high ratio means the
space is genuinely being used.

Labels also collide across metrics. ``"low"`` is emitted by both
``dimensionality`` (4-10% utilization — unhealthy) and ``intrinsic_dim``
(<5% ratio — unhealthy), and ``"severe"`` by both ``anisotropy`` and
``density``. Classifying a bare label string without knowing which metric
produced it is therefore not possible, and any code that tries will silently
mis-grade at least one metric.

All display and scoring code must go through this module rather than testing
interpretation strings against a local set.
"""

from __future__ import annotations

OK = "ok"
WARN = "warn"
BAD = "bad"

# metric name → {interpretation label: severity}
_SEVERITY: dict[str, dict[str, str]] = {
    "anisotropy": {
        "healthy": OK, "moderate": WARN, "high": BAD, "severe": BAD,
    },
    "dimensionality": {
        "healthy": OK, "moderate": WARN, "low": BAD, "critical": BAD,
    },
    "density": {
        "uniform": OK, "moderate": WARN, "clustered": BAD, "severe": BAD,
    },
    "sensitivity": {
        "stable": OK, "moderate": WARN, "sensitive": BAD, "brittle": BAD,
    },
    # Inverted polarity: only a collapsed manifold ("low") is a problem.
    "intrinsic_dim": {
        "very_high": OK, "high": OK, "moderate": OK, "low": BAD,
    },
}

# The metric names carried by AuditReport, in report/display order.
METRIC_NAMES = (
    "anisotropy",
    "dimensionality",
    "density",
    "sensitivity",
    "intrinsic_dim",
)


def severity_of(metric: str, interpretation: str) -> str:
    """
    Return ``OK``/``WARN``/``BAD`` for *interpretation* as produced by *metric*.

    An unknown metric or label degrades to ``BAD`` rather than silently
    reporting healthy — a missing entry should surface, not hide a problem.
    """
    return _SEVERITY.get(metric, {}).get(interpretation, BAD)


def severity(result) -> str:
    """Severity of a metric result dataclass (uses its ``METRIC`` attribute)."""
    return severity_of(getattr(result, "METRIC", ""), result.interpretation)


def is_healthy(result) -> bool:
    """True if *result* indicates no problem for its own metric."""
    return severity(result) == OK
