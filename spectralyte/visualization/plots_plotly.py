"""
spectralyte/visualization/plots.py
=====================================
Visualization layer for Spectralyte audit results.

Produces six publication-quality matplotlib figures:
    1. Anisotropy      — eigenvalue spectrum of the Gram matrix
    2. Dimensionality  — scree plot with cumulative variance curve
    3. Density         — k-NN distance histogram
    4. Sensitivity     — per-embedding stability score histogram
    5. Intrinsic Dim   — TwoNN log-log plot with regression line
    6. Summary         — combined health dashboard (radar + score cards)

Design principles:
    - Each plot is readable by someone unfamiliar with the metric
    - Reference lines show healthy vs unhealthy thresholds
    - Color coding is consistent: green=healthy, amber=moderate, red=severe
    - All plots use a clean, minimal style inspired by modern dev tools
    - Plotly is used instead of matplotlib so plots render natively
      in web contexts (Spectralyte Studio desktop app)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spectralyte.core.report import AuditReport

# Colors — consistent with Spectralyte brand
_C = {
    "navy":   "#0F2744",
    "blue":   "#1A5FA8",
    "teal":   "#0D9488",
    "green":  "#16A34A",
    "amber":  "#D97706",
    "red":    "#DC2626",
    "light":  "#EFF6FF",
    "border": "#CBD5E1",
    "muted":  "#64748B",
    "dark":   "#0F172A",
    "white":  "#FFFFFF",
    "bg":     "#F8FAFC",
}

def _severity_color(interpretation: str) -> str:
    """Map interpretation string to a hex color."""
    healthy = {"healthy", "uniform", "stable", "low"}
    moderate = {"moderate"}
    if interpretation in healthy:
        return _C["green"]
    elif interpretation in moderate:
        return _C["amber"]
    else:
        return _C["red"]


def _require_plotly():
    """Import plotly, raising a clear error if not installed."""
    try:
        import plotly.graph_objects as go
        import plotly.subplots as sp
        return go, sp
    except ImportError:
        raise ImportError(
            "Plotly is required for Spectralyte visualizations. "
            "Install it with: pip install plotly"
        )


# ── Individual metric plots ────────────────────────────────────────────────────

def plot_anisotropy(report: "AuditReport", show: bool = True, save_path: Optional[str] = None):
    """
    Plot the eigenvalue spectrum of the Gram matrix.

    In a healthy isotropic space eigenvalues are approximately equal —
    variance is spread evenly across all directions. In an anisotropic
    space a few eigenvalues dominate — variance is concentrated in a
    small number of directions, which degrades cosine similarity.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, display the plot. Default True.
    save_path : Optional[str]
        If provided, save the figure to this path (.html or .png).
    """
    go, _ = _require_plotly()

    result = report.anisotropy
    eigenvalues = result.eigenvalues
    n_show = min(50, len(eigenvalues))
    evs = eigenvalues[:n_show]
    indices = list(range(1, n_show + 1))

    color = _severity_color(result.interpretation)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=indices,
        y=evs,
        marker_color=color,
        marker_line_color=_C["navy"],
        marker_line_width=0.5,
        name="Eigenvalue",
        hovertemplate="Component %{x}<br>Eigenvalue: %{y:.4f}<extra></extra>",
    ))

    # Reference line — equal eigenvalue baseline
    equal_ev = float(sum(evs) / len(evs))
    fig.add_hline(
        y=equal_ev,
        line_dash="dash",
        line_color=_C["teal"],
        line_width=1.5,
        annotation_text="Equal distribution (isotropic)",
        annotation_position="top right",
        annotation_font_color=_C["teal"],
    )

    fig.update_layout(
        title=dict(
            text=f"Anisotropy — Eigenvalue Spectrum<br>"
                 f"<sup>Score: {result.score:.3f} | {result.interpretation.upper()} | "
                 f"{result.n_vectors:,} vectors × {result.n_dims} dims</sup>",
            font=dict(size=16, color=_C["navy"]),
        ),
        xaxis=dict(
            title="Principal Component (sorted by eigenvalue)",
            tickfont=dict(size=11),
            gridcolor=_C["border"],
        ),
        yaxis=dict(
            title="Eigenvalue",
            tickfont=dict(size=11),
            gridcolor=_C["border"],
        ),
        plot_bgcolor=_C["white"],
        paper_bgcolor=_C["bg"],
        showlegend=False,
        margin=dict(t=100, b=60, l=70, r=40),
        annotations=[dict(
            x=0.02, y=0.97, xref="paper", yref="paper",
            text=f"Concentration in few eigenvalues indicates anisotropy.<br>"
                 f"Flat spectrum = isotropic (healthy). Spike = anisotropic (problematic).",
            showarrow=False,
            font=dict(size=11, color=_C["muted"]),
            align="left",
            bgcolor=_C["light"],
            bordercolor=_C["border"],
            borderwidth=1,
        )]
    )

    _output(fig, show, save_path)
    return fig


def plot_dimensionality(report: "AuditReport", show: bool = True, save_path: Optional[str] = None):
    """
    Plot the scree plot — cumulative explained variance vs number of components.

    Shows how many principal components are needed to explain the variance
    in the embedding space. The vertical line marks the effective dimensionality
    (where cumulative variance crosses the threshold, default 95%).

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, display the plot.
    save_path : Optional[str]
        If provided, save the figure to this path.
    """
    go, sp = _require_plotly()
    from plotly.subplots import make_subplots

    result = report.dimensionality
    cum_var = result.cumulative_variance
    evr = result.explained_variance_ratio
    n_components = len(cum_var)
    x = list(range(1, n_components + 1))
    color = _severity_color(result.interpretation)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Cumulative Explained Variance", "Per-Component Variance"),
        horizontal_spacing=0.12,
    )

    # Left — cumulative variance
    fig.add_trace(go.Scatter(
        x=x, y=list(cum_var * 100),
        mode="lines",
        line=dict(color=_C["blue"], width=2),
        fill="tozeroy",
        fillcolor=f"rgba(26, 95, 168, 0.1)",
        name="Cumulative variance",
        hovertemplate="Component %{x}<br>Cumulative: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    # Threshold line
    threshold_pct = result.variance_threshold * 100
    fig.add_hline(
        y=threshold_pct,
        line_dash="dash", line_color=_C["amber"], line_width=1.5,
        row=1, col=1,
    )

    # Effective dimensionality vertical line
    fig.add_vline(
        x=result.effective_dims,
        line_dash="dot", line_color=color, line_width=2,
        row=1, col=1,
        annotation_text=f"d_eff = {result.effective_dims}",
        annotation_position="top right",
        annotation_font_color=color,
    )

    # Right — per-component variance (bar chart, top 30)
    n_bar = min(30, n_components)
    fig.add_trace(go.Bar(
        x=list(range(1, n_bar + 1)),
        y=list(evr[:n_bar] * 100),
        marker_color=_C["blue"],
        marker_line_width=0,
        name="Per-component",
        hovertemplate="Component %{x}<br>Variance: %{y:.2f}%<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"Effective Dimensionality — Scree Plot<br>"
                 f"<sup>{result.effective_dims} / {result.nominal_dims} dims "
                 f"({result.utilization:.1%} utilization) | "
                 f"PR={result.participation_ratio:.1f} | "
                 f"{result.interpretation.upper()}</sup>",
            font=dict(size=16, color=_C["navy"]),
        ),
        plot_bgcolor=_C["white"],
        paper_bgcolor=_C["bg"],
        showlegend=False,
        margin=dict(t=110, b=60, l=70, r=40),
    )
    fig.update_xaxes(title_text="Number of Components", gridcolor=_C["border"])
    fig.update_yaxes(title_text="Cumulative Variance (%)", gridcolor=_C["border"], row=1, col=1)
    fig.update_yaxes(title_text="Variance Explained (%)", gridcolor=_C["border"], row=1, col=2)

    _output(fig, show, save_path)
    return fig


def plot_density(report: "AuditReport", show: bool = True, save_path: Optional[str] = None):
    """
    Plot the k-NN distance distribution and LOF score distribution.

    Left panel: histogram of k-NN distances. A tight bell curve indicates
    uniform distribution. A wide or bimodal distribution indicates clustering
    with voids — some embeddings are tightly packed, others are isolated.

    Right panel: histogram of LOF scores. Points near 1.0 are normal.
    Points significantly above 1.0 are in anomalously sparse regions.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, display the plot.
    save_path : Optional[str]
        If provided, save the figure to this path.
    """
    go, _ = _require_plotly()
    from plotly.subplots import make_subplots
    import numpy as np

    result = report.density
    color = _severity_color(result.interpretation)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"k-NN Distance Distribution (k={result.k})",
            f"Local Outlier Factor Distribution"
        ),
        horizontal_spacing=0.12,
    )

    # Left — k-NN distance histogram
    fig.add_trace(go.Histogram(
        x=list(result.knn_distances),
        nbinsx=50,
        marker_color=color,
        marker_line_color=_C["navy"],
        marker_line_width=0.3,
        opacity=0.85,
        name="k-NN distances",
        hovertemplate="Distance: %{x:.4f}<br>Count: %{y}<extra></extra>",
    ), row=1, col=1)

    # Mean line
    fig.add_vline(
        x=result.mean_knn_distance,
        line_dash="dash", line_color=_C["navy"], line_width=1.5,
        row=1, col=1,
        annotation_text=f"mean={result.mean_knn_distance:.4f}",
        annotation_position="top right",
        annotation_font_color=_C["navy"],
    )

    # Right — LOF score histogram
    fig.add_trace(go.Histogram(
        x=list(result.lof_scores),
        nbinsx=50,
        marker_color=_C["blue"],
        marker_line_color=_C["navy"],
        marker_line_width=0.3,
        opacity=0.85,
        name="LOF scores",
        hovertemplate="LOF score: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ), row=1, col=2)

    # LOF threshold line
    fig.add_vline(
        x=result.lof_threshold,
        line_dash="dash", line_color=_C["red"], line_width=1.5,
        row=1, col=2,
        annotation_text=f"outlier threshold={result.lof_threshold}",
        annotation_position="top right",
        annotation_font_color=_C["red"],
    )

    fig.update_layout(
        title=dict(
            text=f"Density Distribution<br>"
                 f"<sup>CV={result.cv:.3f} | {result.interpretation.upper()} | "
                 f"{result.n_outliers} outliers detected</sup>",
            font=dict(size=16, color=_C["navy"]),
        ),
        plot_bgcolor=_C["white"],
        paper_bgcolor=_C["bg"],
        showlegend=False,
        margin=dict(t=110, b=60, l=70, r=40),
    )
    fig.update_xaxes(gridcolor=_C["border"])
    fig.update_yaxes(title_text="Count", gridcolor=_C["border"])
    fig.update_xaxes(title_text="Distance to k-th Nearest Neighbor", row=1, col=1)
    fig.update_xaxes(title_text="LOF Score", row=1, col=2)

    _output(fig, show, save_path)
    return fig


def plot_sensitivity(report: "AuditReport", show: bool = True, save_path: Optional[str] = None):
    """
    Plot the per-embedding stability score distribution.

    Histogram of RSI stability scores across all embeddings.
    A healthy space has scores skewed toward 1.0.
    A brittle space has a significant tail toward 0.0.
    The vertical line shows the brittle threshold.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, display the plot.
    save_path : Optional[str]
        If provided, save the figure to this path.
    """
    go, _ = _require_plotly()

    result = report.sensitivity
    color = _severity_color(result.interpretation)
    scores = list(result.stability_per_embedding)

    fig = go.Figure()

    # Stability score histogram
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=40,
        marker_color=color,
        marker_line_color=_C["navy"],
        marker_line_width=0.3,
        opacity=0.85,
        name="Stability scores",
        hovertemplate="Stability: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))

    # Brittle threshold line
    fig.add_vline(
        x=result.brittle_threshold,
        line_dash="dash",
        line_color=_C["red"],
        line_width=2,
        annotation_text=f"Brittle threshold = {result.brittle_threshold}",
        annotation_position="top left",
        annotation_font_color=_C["red"],
    )

    # Mean stability line
    fig.add_vline(
        x=result.mean_stability,
        line_dash="dot",
        line_color=_C["navy"],
        line_width=1.5,
        annotation_text=f"Mean = {result.mean_stability:.3f}",
        annotation_position="top right",
        annotation_font_color=_C["navy"],
    )

    # Shaded brittle region
    fig.add_vrect(
        x0=0, x1=result.brittle_threshold,
        fillcolor=_C["red"],
        opacity=0.05,
        layer="below",
        line_width=0,
    )

    fig.update_layout(
        title=dict(
            text=f"Retrieval Sensitivity Index — Stability Distribution<br>"
                 f"<sup>Mean stability: {result.mean_stability:.3f} | "
                 f"{result.interpretation.upper()} | "
                 f"{result.n_brittle} brittle embeddings "
                 f"({result.brittle_fraction:.1%} of index)</sup>",
            font=dict(size=16, color=_C["navy"]),
        ),
        xaxis=dict(
            title="Stability Score (Jaccard similarity under perturbation)",
            range=[0, 1],
            tickfont=dict(size=11),
            gridcolor=_C["border"],
        ),
        yaxis=dict(
            title="Number of Embeddings",
            tickfont=dict(size=11),
            gridcolor=_C["border"],
        ),
        plot_bgcolor=_C["white"],
        paper_bgcolor=_C["bg"],
        showlegend=False,
        margin=dict(t=110, b=60, l=70, r=40),
        annotations=[dict(
            x=result.brittle_threshold / 2,
            y=1.0,
            yref="paper",
            text="Brittle zone",
            showarrow=False,
            font=dict(size=11, color=_C["red"]),
        )]
    )

    _output(fig, show, save_path)
    return fig


def plot_intrinsic_dim(report: "AuditReport", show: bool = True, save_path: Optional[str] = None):
    """
    Plot the TwoNN log-log fit for intrinsic dimensionality estimation.

    X axis: log(mu) where mu = dist(second NN) / dist(first NN)
    Y axis: log(1 - F(mu)) where F is the empirical CDF of mu values

    Under the manifold assumption this should be a straight line with
    slope = -d_int. The R² of the fit measures how cleanly the data
    conforms to the manifold assumption.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, display the plot.
    save_path : Optional[str]
        If provided, save the figure to this path.
    """
    go, _ = _require_plotly()
    import numpy as np

    result = report.intrinsic_dim
    color = _severity_color(result.interpretation)

    log_mu = list(result.log_mu)
    log_surv = list(result.log_survival)

    # Regression line points
    x_line = [min(log_mu), max(log_mu)]
    y_line = [result.slope * x + result.intercept for x in x_line]

    fig = go.Figure()

    # Scatter of actual data points
    fig.add_trace(go.Scatter(
        x=log_mu,
        y=log_surv,
        mode="markers",
        marker=dict(
            color=_C["blue"],
            size=4,
            opacity=0.6,
        ),
        name="Observed",
        hovertemplate="log(μ): %{x:.3f}<br>log(1-F): %{y:.3f}<extra></extra>",
    ))

    # Regression line
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        line=dict(color=color, width=2.5, dash="solid"),
        name=f"Fit: slope = {result.slope:.3f}",
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(
            text=f"Intrinsic Dimensionality — TwoNN Log-Log Fit<br>"
                 f"<sup>d_int = {result.d_int:.1f} | R² = {result.r_squared:.3f} | "
                 f"Nominal dims = {result.n_dims} | "
                 f"{result.interpretation.upper()}</sup>",
            font=dict(size=16, color=_C["navy"]),
        ),
        xaxis=dict(
            title="log(μ)  where μ = dist(2nd NN) / dist(1st NN)",
            tickfont=dict(size=11),
            gridcolor=_C["border"],
        ),
        yaxis=dict(
            title="log(1 − F(μ))  where F is the empirical CDF",
            tickfont=dict(size=11),
            gridcolor=_C["border"],
        ),
        plot_bgcolor=_C["white"],
        paper_bgcolor=_C["bg"],
        legend=dict(
            x=0.02, y=0.05,
            bgcolor=_C["light"],
            bordercolor=_C["border"],
            borderwidth=1,
        ),
        margin=dict(t=110, b=60, l=80, r=40),
        annotations=[dict(
            x=0.98, y=0.97, xref="paper", yref="paper",
            text=f"Slope = {result.slope:.3f}<br>"
                 f"d_int = {result.d_int:.1f}<br>"
                 f"R² = {result.r_squared:.3f}",
            showarrow=False,
            font=dict(size=12, color=_C["navy"]),
            align="right",
            bgcolor=_C["light"],
            bordercolor=_C["border"],
            borderwidth=1,
        )]
    )

    _output(fig, show, save_path)
    return fig


def plot_summary(report: "AuditReport", show: bool = True, save_path: Optional[str] = None):
    """
    Plot a combined health dashboard showing all five metrics at once.

    Left panel: radar chart showing all five metric scores normalized
    to [0, 1] where 1 is always healthy (scores are inverted where
    lower is better, e.g. anisotropy).

    Right panel: score cards with the raw value and interpretation
    for each metric, color coded by severity.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, display the plot.
    save_path : Optional[str]
        If provided, save the figure to this path.
    """
    go, _ = _require_plotly()
    from plotly.subplots import make_subplots
    import numpy as np

    # ── Normalize scores to [0, 1] where 1 = healthy ──────────────────────────
    # Anisotropy: 0 = best, 1 = worst → invert
    aniso_health = max(0.0, 1.0 - report.anisotropy.score)

    # Utilization: higher = healthier, cap at 1
    dim_health = min(1.0, report.dimensionality.utilization / 0.20)

    # Density CV: 0 = best → invert and normalize
    density_health = max(0.0, 1.0 - min(report.density.cv, 1.0))

    # Stability: higher = healthier
    stability_health = report.sensitivity.mean_stability

    # Intrinsic dim ratio: moderate = healthiest, cap
    id_ratio = report.intrinsic_dim.d_int / report.intrinsic_dim.n_dims
    intrinsic_health = min(1.0, max(0.0, 1.0 - abs(id_ratio - 0.10) / 0.10))

    health_scores = [aniso_health, dim_health, density_health, stability_health, intrinsic_health]
    categories = ["Anisotropy", "Dimensionality", "Density", "Stability", "Intrinsic Dim"]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.45, 0.55],
        specs=[[{"type": "polar"}, {"type": "table"}]],
        subplot_titles=("Geometric Health Radar", "Metric Scores"),
        horizontal_spacing=0.08,
    )

    # ── Radar chart ────────────────────────────────────────────────────────────
    radar_categories = categories + [categories[0]]
    radar_values = health_scores + [health_scores[0]]

    overall_health = float(np.mean(health_scores))
    radar_color = _severity_color(
        "healthy" if overall_health > 0.7
        else "moderate" if overall_health > 0.4
        else "severe"
    )

    fig.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_categories,
        fill="toself",
        fillcolor=f"rgba({_hex_to_rgb(radar_color)}, 0.15)",
        line=dict(color=radar_color, width=2),
        name="Health scores",
    ), row=1, col=1)

    # Healthy threshold ring at 0.7
    fig.add_trace(go.Scatterpolar(
        r=[0.7] * len(radar_categories),
        theta=radar_categories,
        mode="lines",
        line=dict(color=_C["teal"], width=1, dash="dash"),
        name="Healthy threshold",
    ), row=1, col=1)

    fig.update_polars(
        radialaxis=dict(
            visible=True, range=[0, 1],
            tickfont=dict(size=9),
            gridcolor=_C["border"],
        ),
        angularaxis=dict(tickfont=dict(size=11)),
    )

    # ── Score cards table ──────────────────────────────────────────────────────
    metrics = [
        ("Anisotropy", f"{report.anisotropy.score:.3f}", report.anisotropy.interpretation),
        ("Effective Dims", f"{report.dimensionality.effective_dims}/{report.dimensionality.nominal_dims} ({report.dimensionality.utilization:.1%})", report.dimensionality.interpretation),
        ("Density CV", f"{report.density.cv:.3f}", report.density.interpretation),
        ("RSI Stability", f"{report.sensitivity.mean_stability:.3f}", report.sensitivity.interpretation),
        ("Intrinsic Dim", f"{report.intrinsic_dim.d_int:.1f} (R²={report.intrinsic_dim.r_squared:.2f})", report.intrinsic_dim.interpretation),
    ]

    metric_names = [m[0] for m in metrics]
    metric_values = [m[1] for m in metrics]
    metric_interps = [m[2].upper() for m in metrics]
    metric_colors = [_severity_color(m[2]) for m in metrics]

    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>", "<b>Status</b>"],
            fill_color=_C["navy"],
            font=dict(color=_C["white"], size=12),
            align="left",
            height=32,
        ),
        cells=dict(
            values=[metric_names, metric_values, metric_interps],
            fill_color=[
                [_C["light"]] * 5,
                [_C["white"]] * 5,
                metric_colors,
            ],
            font=dict(
                color=[[_C["dark"]] * 5, [_C["dark"]] * 5, [_C["white"]] * 5],
                size=12,
            ),
            align="left",
            height=30,
        ),
    ), row=1, col=2)

    n, d = report.embeddings_shape
    overall_label = (
        "HEALTHY" if overall_health > 0.7
        else "MODERATE" if overall_health > 0.4
        else "NEEDS ATTENTION"
    )

    fig.update_layout(
        title=dict(
            text=f"Spectralyte Audit Summary — {overall_label}<br>"
                 f"<sup>{n:,} vectors × {d} dims | "
                 f"{report.n_issues} issue{'s' if report.n_issues != 1 else ''} detected | "
                 f"Overall health: {overall_health:.0%}</sup>",
            font=dict(size=16, color=_C["navy"]),
        ),
        paper_bgcolor=_C["bg"],
        showlegend=False,
        margin=dict(t=110, b=40, l=40, r=40),
        height=500,
    )

    _output(fig, show, save_path)
    return fig


def plot_all(report: "AuditReport", save_dir: Optional[str] = None, show: bool = True):
    """
    Generate all six Spectralyte visualizations.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    save_dir : Optional[str]
        If provided, save all figures as HTML files in this directory.
    show : bool
        If True, display each plot. Default True.

    Returns
    -------
    dict
        Dictionary mapping metric name to Plotly figure object.
    """
    import os

    def save_path(name):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            return os.path.join(save_dir, f"spectralyte_{name}.html")
        return None

    return {
        "summary":       plot_summary(report, show=show, save_path=save_path("summary")),
        "anisotropy":    plot_anisotropy(report, show=show, save_path=save_path("anisotropy")),
        "dimensionality":plot_dimensionality(report, show=show, save_path=save_path("dimensionality")),
        "density":       plot_density(report, show=show, save_path=save_path("density")),
        "sensitivity":   plot_sensitivity(report, show=show, save_path=save_path("sensitivity")),
        "intrinsic_dim": plot_intrinsic_dim(report, show=show, save_path=save_path("intrinsic_dim")),
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _output(fig, show: bool, save_path: Optional[str]) -> None:
    """Show and/or save a Plotly figure."""
    if save_path:
        if save_path.endswith(".png"):
            try:
                fig.write_image(save_path)
            except Exception:
                html_path = save_path.replace(".png", ".html")
                fig.write_html(html_path)
        else:
            fig.write_html(save_path)
    if show:
        fig.show()


def _hex_to_rgb(hex_color: str) -> str:
    """Convert hex color to 'r, g, b' string for rgba() usage."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r}, {g}, {b}"