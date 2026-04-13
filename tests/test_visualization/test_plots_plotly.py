"""
tests/test_visualization/test_plots.py
========================================
Tests for the Spectralyte visualization layer.

Since these tests run in a headless environment we test that:
- Each plot function returns a Plotly figure object
- Figures have the expected structure (traces, layout, titles)
- plot_all() returns all six figures
- Saving to HTML works correctly
- Figures don't raise on edge case inputs
"""

import numpy as np
import pytest
import tempfile
import os

from spectralyte import Spectralyte
from spectralyte.visualization.plots_plotly import (
    plot_anisotropy, plot_dimensionality, plot_density,
    plot_sensitivity, plot_intrinsic_dim, plot_summary, plot_all,
    _hex_to_rgb, _severity_color
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def report():
    """Pre-computed report for all visualization tests."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 64)
    audit = Spectralyte(embeddings, k=5, random_seed=42)
    return audit.run(verbose=False)


# ── Individual plot tests ──────────────────────────────────────────────────────

def test_plot_anisotropy_returns_figure(report):
    """plot_anisotropy should return a Plotly Figure."""
    import plotly.graph_objects as go
    fig = plot_anisotropy(report, show=False)
    assert isinstance(fig, go.Figure)


def test_plot_anisotropy_has_bar_trace(report):
    """Anisotropy plot should have a Bar trace."""
    fig = plot_anisotropy(report, show=False)
    trace_types = [type(t).__name__ for t in fig.data]
    assert "Bar" in trace_types


def test_plot_anisotropy_title_contains_score(report):
    """Anisotropy plot title should mention the score."""
    fig = plot_anisotropy(report, show=False)
    assert "Anisotropy" in fig.layout.title.text


def test_plot_dimensionality_returns_figure(report):
    """plot_dimensionality should return a Plotly Figure."""
    import plotly.graph_objects as go
    fig = plot_dimensionality(report, show=False)
    assert isinstance(fig, go.Figure)


def test_plot_dimensionality_has_two_subplots(report):
    """Dimensionality plot should have scatter and bar traces."""
    fig = plot_dimensionality(report, show=False)
    assert len(fig.data) >= 2


def test_plot_dimensionality_title_contains_dims(report):
    """Dimensionality plot title should mention effective dims."""
    fig = plot_dimensionality(report, show=False)
    assert "Dimensionality" in fig.layout.title.text


def test_plot_density_returns_figure(report):
    """plot_density should return a Plotly Figure."""
    import plotly.graph_objects as go
    fig = plot_density(report, show=False)
    assert isinstance(fig, go.Figure)


def test_plot_density_has_histogram_traces(report):
    """Density plot should have Histogram traces."""
    fig = plot_density(report, show=False)
    trace_types = [type(t).__name__ for t in fig.data]
    assert "Histogram" in trace_types


def test_plot_density_title_contains_cv(report):
    """Density plot title should mention CV."""
    fig = plot_density(report, show=False)
    assert "Density" in fig.layout.title.text


def test_plot_sensitivity_returns_figure(report):
    """plot_sensitivity should return a Plotly Figure."""
    import plotly.graph_objects as go
    fig = plot_sensitivity(report, show=False)
    assert isinstance(fig, go.Figure)


def test_plot_sensitivity_has_histogram(report):
    """Sensitivity plot should have a Histogram trace."""
    fig = plot_sensitivity(report, show=False)
    trace_types = [type(t).__name__ for t in fig.data]
    assert "Histogram" in trace_types


def test_plot_sensitivity_title_contains_stability(report):
    """Sensitivity plot title should mention stability."""
    fig = plot_sensitivity(report, show=False)
    assert "Sensitivity" in fig.layout.title.text


def test_plot_intrinsic_dim_returns_figure(report):
    """plot_intrinsic_dim should return a Plotly Figure."""
    import plotly.graph_objects as go
    fig = plot_intrinsic_dim(report, show=False)
    assert isinstance(fig, go.Figure)


def test_plot_intrinsic_dim_has_scatter_and_line(report):
    """Intrinsic dim plot should have scatter and line traces."""
    fig = plot_intrinsic_dim(report, show=False)
    assert len(fig.data) >= 2


def test_plot_intrinsic_dim_title_contains_d_int(report):
    """Intrinsic dim plot title should mention d_int."""
    fig = plot_intrinsic_dim(report, show=False)
    assert "Intrinsic" in fig.layout.title.text


def test_plot_summary_returns_figure(report):
    """plot_summary should return a Plotly Figure."""
    import plotly.graph_objects as go
    fig = plot_summary(report, show=False)
    assert isinstance(fig, go.Figure)


def test_plot_summary_has_radar_and_table(report):
    """Summary plot should have polar (radar) and table traces."""
    fig = plot_summary(report, show=False)
    trace_types = [type(t).__name__ for t in fig.data]
    assert "Scatterpolar" in trace_types
    assert "Table" in trace_types


def test_plot_summary_title_contains_audit(report):
    """Summary plot title should mention audit."""
    fig = plot_summary(report, show=False)
    assert "Spectralyte" in fig.layout.title.text


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
    expected_keys = {
        "summary", "anisotropy", "dimensionality",
        "density", "sensitivity", "intrinsic_dim"
    }
    assert set(result.keys()) == expected_keys


def test_plot_all_saves_html_files(report):
    """plot_all with save_dir should create HTML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_all(report, save_dir=tmpdir, show=False)
        files = os.listdir(tmpdir)
        html_files = [f for f in files if f.endswith(".html")]
        assert len(html_files) == 6


def test_plot_all_html_files_not_empty(report):
    """Saved HTML files should not be empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_all(report, save_dir=tmpdir, show=False)
        for fname in os.listdir(tmpdir):
            fpath = os.path.join(tmpdir, fname)
            assert os.path.getsize(fpath) > 1000, f"{fname} is suspiciously small"


# ── Save tests ─────────────────────────────────────────────────────────────────

def test_save_single_plot_to_html(report):
    """Individual plot should save to HTML file."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = f.name
    try:
        plot_anisotropy(report, show=False, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000
    finally:
        os.unlink(path)


# ── Helper function tests ──────────────────────────────────────────────────────

def test_hex_to_rgb_correct():
    """_hex_to_rgb should correctly convert hex to r, g, b string."""
    result = _hex_to_rgb("#1A5FA8")
    assert result == "26, 95, 168"


def test_hex_to_rgb_without_hash():
    """_hex_to_rgb should work with or without leading #."""
    result = _hex_to_rgb("FF0000")
    assert result == "255, 0, 0"


def test_severity_color_healthy():
    """Healthy interpretations should return green."""
    assert _severity_color("healthy") == "#16A34A"
    assert _severity_color("uniform") == "#16A34A"
    assert _severity_color("stable") == "#16A34A"


def test_severity_color_moderate():
    """Moderate interpretation should return amber."""
    assert _severity_color("moderate") == "#D97706"


def test_severity_color_severe():
    """Non-healthy, non-moderate interpretations should return red."""
    assert _severity_color("severe") == "#DC2626"
    assert _severity_color("clustered") == "#DC2626"
    assert _severity_color("brittle") == "#DC2626"