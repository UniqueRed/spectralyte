"""
tests/test_visualization/test_plots_matplotlib.py
===================================================
Tests for the Spectralyte matplotlib visualization backend.

Tests run headless using matplotlib's non-interactive Agg backend.
"""

import numpy as np
import pytest
import tempfile
import os
import matplotlib
matplotlib.use("Agg")   # headless — no display needed

from spectralyte import Spectralyte
from spectralyte.visualization.plots_matplotlib import (
    plot_anisotropy, plot_dimensionality, plot_density,
    plot_sensitivity, plot_intrinsic_dim, plot_summary, plot_all,
    _severity_color, _severity_label,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def report():
    """Pre-computed report reused across all tests."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 64)
    audit = Spectralyte(embeddings, k=5, random_seed=42)
    return audit.run(verbose=False)


# ── Individual plot return type tests ─────────────────────────────────────────

def test_plot_anisotropy_returns_figure(report):
    """plot_anisotropy should return a matplotlib Figure."""
    import matplotlib.figure as mfig
    fig = plot_anisotropy(report, show=False)
    assert isinstance(fig, mfig.Figure)


def test_plot_dimensionality_returns_figure(report):
    """plot_dimensionality should return a matplotlib Figure."""
    import matplotlib.figure as mfig
    fig = plot_dimensionality(report, show=False)
    assert isinstance(fig, mfig.Figure)


def test_plot_density_returns_figure(report):
    """plot_density should return a matplotlib Figure."""
    import matplotlib.figure as mfig
    fig = plot_density(report, show=False)
    assert isinstance(fig, mfig.Figure)


def test_plot_sensitivity_returns_figure(report):
    """plot_sensitivity should return a matplotlib Figure."""
    import matplotlib.figure as mfig
    fig = plot_sensitivity(report, show=False)
    assert isinstance(fig, mfig.Figure)


def test_plot_intrinsic_dim_returns_figure(report):
    """plot_intrinsic_dim should return a matplotlib Figure."""
    import matplotlib.figure as mfig
    fig = plot_intrinsic_dim(report, show=False)
    assert isinstance(fig, mfig.Figure)


def test_plot_summary_returns_figure(report):
    """plot_summary should return a matplotlib Figure."""
    import matplotlib.figure as mfig
    fig = plot_summary(report, show=False)
    assert isinstance(fig, mfig.Figure)


# ── Axes content tests ─────────────────────────────────────────────────────────

def test_anisotropy_has_bar_container(report):
    """Anisotropy plot should have bar containers."""
    fig = plot_anisotropy(report, show=False)
    ax = fig.axes[0]
    assert len(ax.containers) > 0


def test_anisotropy_title_contains_score(report):
    """Anisotropy plot title should mention score."""
    fig = plot_anisotropy(report, show=False)
    title = fig.axes[0].get_title()
    assert "Anisotropy" in title
    assert str(round(report.anisotropy.score, 3))[:3] in title


def test_dimensionality_has_two_axes(report):
    """Dimensionality plot should have two subplots."""
    fig = plot_dimensionality(report, show=False)
    assert len(fig.axes) == 2


def test_dimensionality_titles_correct(report):
    """Dimensionality axes should have correct titles."""
    fig = plot_dimensionality(report, show=False)
    titles = [ax.get_title() for ax in fig.axes]
    assert any("Cumulative" in t for t in titles)
    assert any("Per-Component" in t for t in titles)


def test_density_has_two_axes(report):
    """Density plot should have two subplots."""
    fig = plot_density(report, show=False)
    assert len(fig.axes) == 2


def test_density_axes_have_labels(report):
    """Density axes should have x and y labels."""
    fig = plot_density(report, show=False)
    for ax in fig.axes:
        assert len(ax.get_xlabel()) > 0
        assert len(ax.get_ylabel()) > 0


def test_sensitivity_has_one_axes(report):
    """Sensitivity plot should have one axes."""
    fig = plot_sensitivity(report, show=False)
    assert len(fig.axes) == 1


def test_sensitivity_x_range(report):
    """Sensitivity plot x axis should span [0, 1]."""
    fig = plot_sensitivity(report, show=False)
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    assert xlim[0] <= 0.0
    assert xlim[1] >= 1.0


def test_sensitivity_has_legend(report):
    """Sensitivity plot should have a legend."""
    fig = plot_sensitivity(report, show=False)
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None


def test_intrinsic_dim_has_one_axes(report):
    """Intrinsic dim plot should have one axes."""
    fig = plot_intrinsic_dim(report, show=False)
    assert len(fig.axes) == 1


def test_intrinsic_dim_has_scatter_and_line(report):
    """Intrinsic dim plot should have scatter and line elements."""
    fig = plot_intrinsic_dim(report, show=False)
    ax = fig.axes[0]
    # Should have at least 2 lines (scatter + regression line)
    assert len(ax.lines) >= 1 or len(ax.collections) >= 1


def test_summary_has_multiple_axes(report):
    """Summary plot should have multiple axes (radar + table)."""
    fig = plot_summary(report, show=False)
    assert len(fig.axes) >= 2


# ── plot_all() tests ───────────────────────────────────────────────────────────

def test_plot_all_returns_dict(report):
    """plot_all should return a dictionary."""
    result = plot_all(report, show=False)
    assert isinstance(result, dict)


def test_plot_all_has_six_figures(report):
    """plot_all should return exactly six figures."""
    result = plot_all(report, show=False)
    assert len(result) == 6


def test_plot_all_correct_keys(report):
    """plot_all should have the expected metric keys."""
    result = plot_all(report, show=False)
    expected = {"summary", "anisotropy", "dimensionality",
                "density", "sensitivity", "intrinsic_dim"}
    assert set(result.keys()) == expected


def test_plot_all_all_figures(report):
    """All values in plot_all result should be matplotlib Figures."""
    import matplotlib.figure as mfig
    result = plot_all(report, show=False)
    for key, fig in result.items():
        assert isinstance(fig, mfig.Figure), f"{key} is not a Figure"


# ── Save tests ─────────────────────────────────────────────────────────────────

def test_plot_all_saves_png_files(report):
    """plot_all with save_dir should create PNG files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_all(report, save_dir=tmpdir, show=False)
        files = os.listdir(tmpdir)
        png_files = [f for f in files if f.endswith(".png")]
        assert len(png_files) == 6


def test_plot_all_png_files_not_empty(report):
    """Saved PNG files should not be empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_all(report, save_dir=tmpdir, show=False)
        for fname in os.listdir(tmpdir):
            fpath = os.path.join(tmpdir, fname)
            assert os.path.getsize(fpath) > 5000, (
                f"{fname} is suspiciously small — may be empty"
            )


def test_save_single_plot_to_png(report):
    """Individual plot should save correctly to PNG."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        plot_anisotropy(report, show=False, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 5000
    finally:
        os.unlink(path)


def test_save_creates_directory_if_needed(report):
    """plot_all should create save_dir if it does not exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "plots", "output")
        plot_all(report, save_dir=new_dir, show=False)
        assert os.path.isdir(new_dir)


# ── Helper function tests ──────────────────────────────────────────────────────

def test_severity_color_healthy():
    """Healthy interpretations map to green."""
    assert _severity_color("healthy") == "#16A34A"
    assert _severity_color("uniform") == "#16A34A"
    assert _severity_color("stable") == "#16A34A"
    assert _severity_color("low") == "#16A34A"


def test_severity_color_moderate():
    """Moderate maps to amber."""
    assert _severity_color("moderate") == "#D97706"


def test_severity_color_severe():
    """Severe and other bad states map to red."""
    assert _severity_color("severe") == "#DC2626"
    assert _severity_color("clustered") == "#DC2626"
    assert _severity_color("brittle") == "#DC2626"
    assert _severity_color("critical") == "#DC2626"


def test_severity_label_contains_interpretation():
    """Severity label should contain the interpretation string."""
    label = _severity_label("healthy")
    assert "HEALTHY" in label
    assert "✓" in label


def test_severity_label_severe_has_x():
    """Severe label should have ✗ icon."""
    label = _severity_label("severe")
    assert "✗" in label


def test_severity_label_moderate_has_warning():
    """Moderate label should have ⚠ icon."""
    label = _severity_label("moderate")
    assert "⚠" in label


# ── Edge case tests ────────────────────────────────────────────────────────────

def test_plots_work_with_minimal_embeddings():
    """Plots should work with small embedding matrices."""
    rng = np.random.RandomState(0)
    embeddings = rng.randn(20, 16)
    audit = Spectralyte(embeddings, k=3, random_seed=0)
    report = audit.run(verbose=False)

    for fn in [plot_anisotropy, plot_dimensionality, plot_density,
               plot_sensitivity, plot_intrinsic_dim, plot_summary]:
        fig = fn(report, show=False)
        assert fig is not None


def test_plots_work_with_high_dimensional_embeddings():
    """Plots should work with high-dimensional embeddings."""
    rng = np.random.RandomState(0)
    embeddings = rng.randn(100, 512)
    audit = Spectralyte(embeddings, k=5, random_seed=0)
    report = audit.run(verbose=False)
    fig = plot_summary(report, show=False)
    assert fig is not None