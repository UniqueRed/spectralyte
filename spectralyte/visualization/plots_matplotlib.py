"""
spectralyte/visualization/plots_matplotlib.py
===============================================
Matplotlib visualization backend for Spectralyte audit results.

Produces six publication-quality figures suitable for use in
Jupyter notebooks, Python scripts, and inline terminal output.
This is the default backend for the Python library.

For the interactive web/desktop backend see plots_plotly.py.

Six figures:
    1. Anisotropy      — eigenvalue spectrum bar chart
    2. Dimensionality  — scree plot with cumulative variance curve
    3. Density         — k-NN distance histogram + LOF distribution
    4. Sensitivity     — per-embedding stability score histogram
    5. Intrinsic Dim   — TwoNN log-log fit with regression line
    6. Summary         — combined health dashboard (radar + score cards)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from spectralyte.core.report import AuditReport

# ── Design tokens ──────────────────────────────────────────────────────────────
_C = {
    "navy":   "#0F2744",
    "blue":   "#1A5FA8",
    "teal":   "#0D9488",
    "green":  "#16A34A",
    "amber":  "#D97706",
    "red":    "#DC2626",
    "light":  "#EFF6FF",
    "border": "#CBD5E1",
    "muted":  "#94A3B8",
    "dark":   "#0F172A",
    "white":  "#FFFFFF",
    "bg":     "#F8FAFC",
}

_STYLE = {
    "figure.facecolor":  _C["bg"],
    "axes.facecolor":    _C["white"],
    "axes.edgecolor":    _C["border"],
    "axes.labelcolor":   _C["dark"],
    "axes.titlecolor":   _C["navy"],
    "axes.grid":         True,
    "grid.color":        _C["border"],
    "grid.linewidth":    0.6,
    "grid.alpha":        0.8,
    "xtick.color":       _C["muted"],
    "ytick.color":       _C["muted"],
    "text.color":        _C["dark"],
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  _C["border"],
}


def _severity_color(interpretation: str) -> str:
    """Map interpretation to hex color."""
    healthy = {"healthy", "uniform", "stable", "low"}
    moderate = {"moderate"}
    if interpretation in healthy:
        return _C["green"]
    elif interpretation in moderate:
        return _C["amber"]
    else:
        return _C["red"]


def _severity_label(interpretation: str) -> str:
    """Map interpretation to display label with icon."""
    icons = {
        "healthy": "✓", "uniform": "✓", "stable": "✓", "low": "✓",
        "moderate": "⚠",
    }
    icon = icons.get(interpretation, "✗")
    return f"{icon}  {interpretation.upper()}"


def _apply_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent styling to an axes object."""
    ax.set_title(title, color=_C["navy"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=_C["dark"], fontsize=10)
    ax.set_ylabel(ylabel, color=_C["dark"], fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_C["border"])
    ax.spines["bottom"].set_color(_C["border"])
    ax.tick_params(colors=_C["muted"])


def _require_matplotlib():
    """Import matplotlib, raising a clear error if not installed."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        return matplotlib, plt, mpatches
    except ImportError:
        raise ImportError(
            "Matplotlib is required for Spectralyte visualizations. "
            "Install it with: pip install matplotlib"
        )


# ── Individual metric plots ────────────────────────────────────────────────────

def plot_anisotropy(
    report: "AuditReport",
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot the eigenvalue spectrum of the Gram matrix.

    In a healthy isotropic space eigenvalues are approximately equal.
    In an anisotropic space a few eigenvalues dominate, indicating
    that variance is concentrated in a small number of directions.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, call plt.show(). Default True.
    save_path : Optional[str]
        If provided, save figure to this path.
    figsize : Tuple[int, int]
        Figure size in inches.
    """
    mpl, plt, mpatches = _require_matplotlib()

    result = report.anisotropy
    eigenvalues = result.eigenvalues
    n_show = min(50, len(eigenvalues))
    evs = eigenvalues[:n_show]
    indices = np.arange(1, n_show + 1)
    color = _severity_color(result.interpretation)

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(indices, evs, color=color, alpha=0.85,
                      edgecolor=_C["navy"], linewidth=0.4, zorder=3)

        # Equal distribution reference line
        equal_ev = float(np.mean(evs))
        ax.axhline(equal_ev, color=_C["teal"], linestyle="--",
                   linewidth=1.5, label=f"Equal distribution (isotropic) = {equal_ev:.3f}",
                   zorder=4)

        _apply_style(ax,
                     title=f"Anisotropy — Eigenvalue Spectrum  "
                           f"[Score: {result.score:.3f}  {_severity_label(result.interpretation)}]",
                     xlabel="Principal Component (sorted descending)",
                     ylabel="Eigenvalue")

        ax.legend(fontsize=9)

        # Annotation box
        ax.text(0.98, 0.97,
                f"n = {result.n_vectors:,}  |  d = {result.n_dims}\n"
                f"Flat spectrum → isotropic (healthy)\n"
                f"Dominant spike → anisotropic (problematic)",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5,
                color=_C["muted"],
                bbox=dict(boxstyle="round,pad=0.4", facecolor=_C["light"],
                          edgecolor=_C["border"], alpha=0.9))

        fig.suptitle("", y=0)
        plt.tight_layout()
        _output_mpl(fig, show, save_path)

    return fig


def plot_dimensionality(
    report: "AuditReport",
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot the scree plot showing cumulative and per-component explained variance.

    Left: cumulative variance with 95% threshold line and effective
    dimensionality marker. Right: per-component variance bar chart
    showing the contribution of each individual component.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, call plt.show().
    save_path : Optional[str]
        If provided, save figure to this path.
    figsize : Tuple[int, int]
        Figure size in inches.
    """
    mpl, plt, mpatches = _require_matplotlib()

    result = report.dimensionality
    cum_var = result.cumulative_variance * 100
    evr = result.explained_variance_ratio * 100
    n = len(cum_var)
    x = np.arange(1, n + 1)
    color = _severity_color(result.interpretation)

    with mpl.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ── Left: cumulative variance ──────────────────────────────────────────
        ax1.fill_between(x, cum_var, alpha=0.15, color=_C["blue"])
        ax1.plot(x, cum_var, color=_C["blue"], linewidth=2, zorder=3)

        threshold_pct = result.variance_threshold * 100
        ax1.axhline(threshold_pct, color=_C["amber"], linestyle="--",
                    linewidth=1.5, label=f"{threshold_pct:.0f}% threshold", zorder=4)

        ax1.axvline(result.effective_dims, color=color, linestyle=":",
                    linewidth=2, label=f"d_eff = {result.effective_dims}", zorder=4)

        ax1.scatter([result.effective_dims], [cum_var[result.effective_dims - 1]],
                    color=color, s=60, zorder=5)

        _apply_style(ax1,
                     title="Cumulative Explained Variance",
                     xlabel="Number of Components",
                     ylabel="Cumulative Variance (%)")
        ax1.set_ylim(0, 105)
        ax1.legend(fontsize=9)

        # ── Right: per-component variance ─────────────────────────────────────
        n_bar = min(30, n)
        ax2.bar(np.arange(1, n_bar + 1), evr[:n_bar],
                color=_C["blue"], alpha=0.85,
                edgecolor=_C["navy"], linewidth=0.3, zorder=3)

        ax2.axvline(result.effective_dims, color=color, linestyle=":",
                    linewidth=2, label=f"d_eff = {result.effective_dims}", zorder=4)

        _apply_style(ax2,
                     title="Per-Component Variance (top 30)",
                     xlabel="Component",
                     ylabel="Variance Explained (%)")
        ax2.legend(fontsize=9)

        fig.suptitle(
            f"Effective Dimensionality  —  "
            f"{result.effective_dims} / {result.nominal_dims} dims used  "
            f"({result.utilization:.1%})  |  PR = {result.participation_ratio:.1f}  |  "
            f"{_severity_label(result.interpretation)}",
            color=_C["navy"], fontsize=12, fontweight="bold", y=1.01
        )

        plt.tight_layout()
        _output_mpl(fig, show, save_path)

    return fig


def plot_density(
    report: "AuditReport",
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot the k-NN distance distribution and LOF score distribution.

    Left: histogram of k-NN distances. Tight bell = uniform.
    Wide/bimodal = clustering with voids.
    Right: LOF score histogram with outlier threshold line.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, call plt.show().
    save_path : Optional[str]
        If provided, save figure to this path.
    figsize : Tuple[int, int]
        Figure size in inches.
    """
    mpl, plt, mpatches = _require_matplotlib()

    result = report.density
    color = _severity_color(result.interpretation)

    with mpl.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # ── Left: k-NN distance histogram ─────────────────────────────────────
        ax1.hist(result.knn_distances, bins=50, color=color, alpha=0.85,
                 edgecolor=_C["navy"], linewidth=0.3, zorder=3)

        ax1.axvline(result.mean_knn_distance, color=_C["navy"],
                    linestyle="--", linewidth=1.5,
                    label=f"Mean = {result.mean_knn_distance:.4f}", zorder=4)

        _apply_style(ax1,
                     title=f"k-NN Distance Distribution  (k={result.k})",
                     xlabel="Distance to k-th Nearest Neighbor",
                     ylabel="Count")

        ax1.text(0.97, 0.97,
                 f"CV = {result.cv:.3f}\n"
                 f"Tight bell = uniform\n"
                 f"Wide spread = clustered",
                 transform=ax1.transAxes,
                 ha="right", va="top", fontsize=8.5, color=_C["muted"],
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=_C["light"],
                           edgecolor=_C["border"], alpha=0.9))
        ax1.legend(fontsize=9)

        # ── Right: LOF score histogram ─────────────────────────────────────────
        ax2.hist(result.lof_scores, bins=50, color=_C["blue"], alpha=0.85,
                 edgecolor=_C["navy"], linewidth=0.3, zorder=3)

        ax2.axvline(result.lof_threshold, color=_C["red"],
                    linestyle="--", linewidth=1.5,
                    label=f"Outlier threshold = {result.lof_threshold}", zorder=4)

        _apply_style(ax2,
                     title="Local Outlier Factor Distribution",
                     xlabel="LOF Score",
                     ylabel="Count")

        ax2.text(0.97, 0.97,
                 f"{result.n_outliers} outliers detected\n"
                 f"LOF ≈ 1.0 → normal\n"
                 f"LOF >> 1.0 → sparse outlier",
                 transform=ax2.transAxes,
                 ha="right", va="top", fontsize=8.5, color=_C["muted"],
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=_C["light"],
                           edgecolor=_C["border"], alpha=0.9))
        ax2.legend(fontsize=9)

        fig.suptitle(
            f"Density Distribution  —  CV = {result.cv:.3f}  |  "
            f"{_severity_label(result.interpretation)}  |  "
            f"{result.n_outliers} outliers",
            color=_C["navy"], fontsize=12, fontweight="bold", y=1.01
        )

        plt.tight_layout()
        _output_mpl(fig, show, save_path)

    return fig


def plot_sensitivity(
    report: "AuditReport",
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot the per-embedding stability score distribution.

    Histogram of RSI stability scores. Healthy space: skewed toward 1.
    Brittle space: significant tail toward 0.
    Vertical lines show the brittle threshold and mean stability.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, call plt.show().
    save_path : Optional[str]
        If provided, save figure to this path.
    figsize : Tuple[int, int]
        Figure size in inches.
    """
    mpl, plt, mpatches = _require_matplotlib()

    result = report.sensitivity
    color = _severity_color(result.interpretation)

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(result.stability_per_embedding, bins=40,
                color=color, alpha=0.85,
                edgecolor=_C["navy"], linewidth=0.3,
                range=(0, 1), zorder=3)

        # Shaded brittle region
        ax.axvspan(0, result.brittle_threshold,
                   color=_C["red"], alpha=0.06, zorder=2,
                   label=f"Brittle zone (< {result.brittle_threshold})")

        # Brittle threshold line
        ax.axvline(result.brittle_threshold, color=_C["red"],
                   linestyle="--", linewidth=2,
                   label=f"Brittle threshold = {result.brittle_threshold}", zorder=4)

        # Mean stability line
        ax.axvline(result.mean_stability, color=_C["navy"],
                   linestyle=":", linewidth=1.5,
                   label=f"Mean stability = {result.mean_stability:.3f}", zorder=4)

        _apply_style(ax,
                     title=f"Retrieval Sensitivity Index — Stability Distribution  "
                           f"[{_severity_label(result.interpretation)}]",
                     xlabel="Stability Score  (Jaccard similarity under perturbation)",
                     ylabel="Number of Embeddings")

        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)

        # Brittle zone label
        ax.text(result.brittle_threshold / 2, ax.get_ylim()[1] * 0.95,
                "Brittle\nzone",
                ha="center", va="top", fontsize=8.5,
                color=_C["red"], alpha=0.8)

        ax.text(0.98, 0.97,
                f"{result.n_brittle} brittle embeddings\n"
                f"({result.brittle_fraction:.1%} of index)\n"
                f"ε = {result.epsilon_fraction} × mean k-NN dist",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5, color=_C["muted"],
                bbox=dict(boxstyle="round,pad=0.4", facecolor=_C["light"],
                          edgecolor=_C["border"], alpha=0.9))

        plt.tight_layout()
        _output_mpl(fig, show, save_path)

    return fig


def plot_intrinsic_dim(
    report: "AuditReport",
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot the TwoNN log-log fit for intrinsic dimensionality estimation.

    X: log(μ) where μ = dist(2nd NN) / dist(1st NN)
    Y: log(1 − F(μ)) where F is the empirical CDF of μ values

    A straight line with slope = −d_int confirms the manifold
    assumption. R² measures fit quality.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, call plt.show().
    save_path : Optional[str]
        If provided, save figure to this path.
    figsize : Tuple[int, int]
        Figure size in inches.
    """
    mpl, plt, mpatches = _require_matplotlib()

    result = report.intrinsic_dim
    color = _severity_color(result.interpretation)

    log_mu = result.log_mu
    log_surv = result.log_survival

    # Regression line
    x_line = np.linspace(log_mu.min(), log_mu.max(), 100)
    y_line = result.slope * x_line + result.intercept

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(log_mu, log_surv,
                   color=_C["blue"], alpha=0.5, s=12,
                   zorder=3, label="Observed μ values")

        ax.plot(x_line, y_line,
                color=color, linewidth=2.5,
                label=f"Fit: slope = {result.slope:.3f}  (d_int = {result.d_int:.1f})",
                zorder=4)

        _apply_style(ax,
                     title=f"Intrinsic Dimensionality — TwoNN Log-Log Fit  "
                           f"[d_int = {result.d_int:.1f}  |  "
                           f"R² = {result.r_squared:.3f}  |  "
                           f"{_severity_label(result.interpretation)}]",
                     xlabel="log(μ)   where  μ = dist(2nd NN) / dist(1st NN)",
                     ylabel="log(1 − F(μ))")

        ax.legend(fontsize=9)

        ax.text(0.97, 0.97,
                f"d_int = {result.d_int:.1f}\n"
                f"Nominal dims = {result.n_dims}\n"
                f"R² = {result.r_squared:.3f}\n"
                f"n points = {result.n_points_used:,}",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=_C["dark"],
                bbox=dict(boxstyle="round,pad=0.5", facecolor=_C["light"],
                          edgecolor=_C["border"], alpha=0.95))

        plt.tight_layout()
        _output_mpl(fig, show, save_path)

    return fig


def plot_summary(
    report: "AuditReport",
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
):
    """
    Combined health dashboard showing all five metrics at once.

    Left: radar chart of normalized health scores (1 = healthy).
    Right: score card table with raw values and color-coded status.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    show : bool
        If True, call plt.show().
    save_path : Optional[str]
        If provided, save figure to this path.
    figsize : Tuple[int, int]
        Figure size in inches.
    """
    mpl, plt, mpatches = _require_matplotlib()

    # ── Normalize health scores to [0, 1] where 1 = healthy ───────────────────
    aniso_health    = max(0.0, 1.0 - report.anisotropy.score)
    dim_health      = min(1.0, report.dimensionality.utilization / 0.20)
    density_health  = max(0.0, 1.0 - min(report.density.cv, 1.0))
    stability_health = report.sensitivity.mean_stability
    id_ratio = report.intrinsic_dim.d_int / max(report.intrinsic_dim.n_dims, 1)
    intrinsic_health = min(1.0, max(0.0, 1.0 - abs(id_ratio - 0.10) / 0.10))

    health_scores = [aniso_health, dim_health, density_health,
                     stability_health, intrinsic_health]
    categories = ["Anisotropy", "Dimensionality", "Density",
                  "Stability", "Intrinsic Dim"]

    overall = float(np.mean(health_scores))

    with mpl.rc_context(_STYLE):
        fig = plt.figure(figsize=figsize, facecolor=_C["bg"])

        # ── Radar chart ────────────────────────────────────────────────────────
        ax_radar = fig.add_subplot(121, polar=True, facecolor=_C["white"])

        n_cats = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        scores_plot = health_scores + [health_scores[0]]
        angles_plot = angles + [angles[0]]

        radar_color = (
            _C["green"] if overall > 0.7
            else _C["amber"] if overall > 0.4
            else _C["red"]
        )

        ax_radar.fill(angles_plot, scores_plot,
                      color=radar_color, alpha=0.15)
        ax_radar.plot(angles_plot, scores_plot,
                      color=radar_color, linewidth=2)

        # Healthy threshold ring
        threshold_vals = [0.7] * (n_cats + 1)
        ax_radar.plot(angles_plot, threshold_vals,
                      color=_C["teal"], linewidth=1,
                      linestyle="--", alpha=0.7, label="Healthy threshold")

        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(categories, fontsize=9, color=_C["dark"])
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_radar.set_yticklabels(["0.25", "0.50", "0.75", "1.0"],
                                  fontsize=7, color=_C["muted"])
        ax_radar.grid(color=_C["border"], linewidth=0.6)
        ax_radar.set_title("Geometric Health Radar",
                           color=_C["navy"], fontsize=11,
                           fontweight="bold", pad=15)

        # Overall score text
        ax_radar.text(0, -0.25,
                      f"Overall: {overall:.0%}",
                      ha="center", va="center",
                      transform=ax_radar.transAxes,
                      fontsize=11, fontweight="bold",
                      color=radar_color)

        # ── Score card table ───────────────────────────────────────────────────
        ax_table = fig.add_subplot(122)
        ax_table.axis("off")

        n, d = report.embeddings_shape
        rows = [
            ["Anisotropy Score",
             f"{report.anisotropy.score:.3f}",
             _severity_label(report.anisotropy.interpretation),
             report.anisotropy.interpretation],
            ["Effective Dimensions",
             f"{report.dimensionality.effective_dims} / {report.dimensionality.nominal_dims}  ({report.dimensionality.utilization:.1%})",
             _severity_label(report.dimensionality.interpretation),
             report.dimensionality.interpretation],
            ["Density CV",
             f"{report.density.cv:.3f}",
             _severity_label(report.density.interpretation),
             report.density.interpretation],
            ["Retrieval Stability",
             f"{report.sensitivity.mean_stability:.3f}",
             _severity_label(report.sensitivity.interpretation),
             report.sensitivity.interpretation],
            ["Intrinsic Dimension",
             f"{report.intrinsic_dim.d_int:.1f}  (R²={report.intrinsic_dim.r_squared:.2f})",
             _severity_label(report.intrinsic_dim.interpretation),
             report.intrinsic_dim.interpretation],
        ]

        col_labels = ["Metric", "Value", "Status"]
        table_data = [[r[0], r[1], r[2]] for r in rows]
        cell_colors = [
            [_C["light"], _C["white"], _severity_color(r[3])]
            for r in rows
        ]

        tbl = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="left",
            loc="center",
            cellColours=cell_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.0, 1.8)

        # Style header
        for j in range(3):
            tbl[(0, j)].set_facecolor(_C["navy"])
            tbl[(0, j)].get_text().set_color(_C["white"])
            tbl[(0, j)].get_text().set_fontweight("bold")

        # White text for status column
        for i in range(1, len(rows) + 1):
            tbl[(i, 2)].get_text().set_color(_C["white"])
            tbl[(i, 2)].get_text().set_fontweight("bold")

        ax_table.set_title("Metric Scores",
                           color=_C["navy"], fontsize=11,
                           fontweight="bold", pad=10)

        overall_label = (
            "HEALTHY" if overall > 0.7
            else "MODERATE" if overall > 0.4
            else "NEEDS ATTENTION"
        )
        fig.suptitle(
            f"Spectralyte Audit Summary — {overall_label}  |  "
            f"{n:,} vectors × {d} dims  |  "
            f"{report.n_issues} issue{'s' if report.n_issues != 1 else ''} detected",
            color=_C["navy"], fontsize=13, fontweight="bold", y=1.01
        )

        plt.tight_layout()
        _output_mpl(fig, show, save_path)

    return fig


def plot_all(
    report: "AuditReport",
    save_dir: Optional[str] = None,
    show: bool = True,
) -> dict:
    """
    Generate all six Spectralyte visualizations.

    Parameters
    ----------
    report : AuditReport
        Results from Spectralyte.run().
    save_dir : Optional[str]
        If provided, save all figures as PNG files in this directory.
    show : bool
        If True, display each plot. Default True.

    Returns
    -------
    dict
        Dictionary mapping metric name to matplotlib Figure object.
    """
    import os

    def save_path(name):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            return os.path.join(save_dir, f"spectralyte_{name}.png")
        return None

    return {
        "summary":        plot_summary(report, show=show, save_path=save_path("summary")),
        "anisotropy":     plot_anisotropy(report, show=show, save_path=save_path("anisotropy")),
        "dimensionality": plot_dimensionality(report, show=show, save_path=save_path("dimensionality")),
        "density":        plot_density(report, show=show, save_path=save_path("density")),
        "sensitivity":    plot_sensitivity(report, show=show, save_path=save_path("sensitivity")),
        "intrinsic_dim":  plot_intrinsic_dim(report, show=show, save_path=save_path("intrinsic_dim")),
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _output_mpl(fig, show: bool, save_path: Optional[str]) -> None:
    """Save and/or display a matplotlib figure."""
    import matplotlib.pyplot as plt
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=_C["bg"])
    if show:
        plt.show()
    else:
        plt.close(fig)