"""
spectralyte/core/report.py
============================
AuditReport — structured results from a full Spectralyte audit.

Wraps the results of all five geometric metrics into a single cohesive
object with human-readable summary, visualization, and export capabilities.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, TYPE_CHECKING

from spectralyte.metrics.anisotropy import AnisotropyResult
from spectralyte.metrics.dimensionality import DimensionalityResult
from spectralyte.metrics.density import DensityResult
from spectralyte.metrics.sensitivity import SensitivityResult
from spectralyte.metrics.intrinsic_dim import IntrinsicDimResult


# ── Severity color codes for terminal output ───────────────────────────────────

_COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


def _colorize(text: str, color: str) -> str:
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def _severity_color(interpretation: str) -> str:
    if interpretation in {"healthy", "uniform", "stable", "low"}:
        return "green"
    elif interpretation in {"moderate"}:
        return "yellow"
    else:
        return "red"


def _severity_icon(interpretation: str) -> str:
    if interpretation in {"healthy", "uniform", "stable", "low"}:
        return "✓"
    elif interpretation in {"moderate"}:
        return "⚠"
    else:
        return "✗"


# ── AuditReport ────────────────────────────────────────────────────────────────

@dataclass
class AuditReport:
    """
    Structured results from a full Spectralyte geometric audit.

    Contains results from all five metrics plus metadata about the audit
    configuration. Provides summary(), plot(), compare(), fix_plan(),
    and export() methods for different use cases.

    Attributes
    ----------
    anisotropy : AnisotropyResult
        Gram matrix anisotropy score and eigenvalue distribution.
    dimensionality : DimensionalityResult
        SVD-based effective dimensionality and participation ratio.
    density : DensityResult
        k-NN distance distribution, CV, and LOF scores.
    sensitivity : SensitivityResult
        Retrieval Sensitivity Index and brittle zone map.
    intrinsic_dim : IntrinsicDimResult
        TwoNN intrinsic dimensionality estimate.
    n_issues : int
        Number of metrics with non-healthy interpretation.
    embeddings_shape : tuple
        Shape (n, d) of the audited embedding matrix.
    """

    anisotropy: AnisotropyResult
    dimensionality: DimensionalityResult
    density: DensityResult
    sensitivity: SensitivityResult
    intrinsic_dim: IntrinsicDimResult
    embeddings_shape: tuple
    _pre_transform_report: Optional["AuditReport"] = field(
        default=None, repr=False
    )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def n_issues(self) -> int:
        """Number of metrics with non-healthy/non-stable interpretation."""
        healthy = {"healthy", "uniform", "stable", "low"}
        interpretations = [
            self.anisotropy.interpretation,
            self.dimensionality.interpretation,
            self.density.interpretation,
            self.sensitivity.interpretation,
            self.intrinsic_dim.interpretation,
        ]
        return sum(1 for i in interpretations if i not in healthy)

    @property
    def needs_transform(self) -> bool:
        """
        True if anisotropy or dimensionality issues detected.
        These are the two problems fixable via direct embedding transforms.
        """
        return self.anisotropy.interpretation not in {"healthy"} or \
               self.dimensionality.interpretation not in {"healthy", "moderate"}

    @property
    def has_brittle_zones(self) -> bool:
        """True if any embeddings are in brittle retrieval zones."""
        return self.sensitivity.n_brittle > 0

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self, use_color: bool = True) -> None:
        """
        Print a human-readable audit summary to stdout.

        Parameters
        ----------
        use_color : bool
            Whether to use terminal color codes. Default True.
            Set to False for plain text output or logging.

        Example
        -------
        >>> report.summary()
        Spectralyte Audit Report
        ════════════════════════════════════════════════════
          Anisotropy Score         0.61   ✗  SEVERE
          Effective Dimensions     43 / 1536  (2.8%)
          ...
        """
        def fmt(text: str, color: str) -> str:
            return _colorize(text, color) if use_color else text

        n, d = self.embeddings_shape
        width = 52

        lines = [
            "",
            fmt("Spectralyte Audit Report", "bold"),
            "═" * width,
            fmt(f"  Embeddings: {n:,} vectors × {d} dims", "bold"),
            "─" * width,
        ]

        # ── Anisotropy ─────────────────────────────────────────────────────────
        a = self.anisotropy
        color = _severity_color(a.interpretation)
        icon = _severity_icon(a.interpretation)
        lines.append(
            f"  Anisotropy Score       "
            f"{fmt(f'{a.score:.3f}', color)}"
            f"   {fmt(icon + '  ' + a.interpretation.upper(), color)}"
        )

        # ── Effective dimensionality ───────────────────────────────────────────
        dm = self.dimensionality
        color = _severity_color(dm.interpretation)
        icon = _severity_icon(dm.interpretation)
        lines.append(
            f"  Effective Dimensions   "
            f"{fmt(f'{dm.effective_dims} / {dm.nominal_dims}', color)}"
            f"  ({dm.utilization:.1%})"
            f"   {fmt(icon + '  ' + dm.interpretation.upper(), color)}"
        )

        # ── Density ────────────────────────────────────────────────────────────
        den = self.density
        color = _severity_color(den.interpretation)
        icon = _severity_icon(den.interpretation)
        lines.append(
            f"  Density CV             "
            f"{fmt(f'{den.cv:.3f}', color)}"
            f"   {fmt(icon + '  ' + den.interpretation.upper(), color)}"
        )

        # ── Sensitivity ────────────────────────────────────────────────────────
        s = self.sensitivity
        color = _severity_color(s.interpretation)
        icon = _severity_icon(s.interpretation)
        lines.append(
            f"  Retrieval Stability    "
            f"{fmt(f'{s.mean_stability:.3f}', color)}"
            f"   {fmt(icon + '  ' + s.interpretation.upper(), color)}"
        )
        if s.n_brittle > 0:
            lines.append(
                f"    └─ {fmt(str(s.n_brittle), 'red')} brittle zone embeddings "
                f"({s.brittle_fraction:.1%} of index)"
            )

        # ── Intrinsic dimensionality ───────────────────────────────────────────
        id_ = self.intrinsic_dim
        color = _severity_color(id_.interpretation)
        icon = _severity_icon(id_.interpretation)
        lines.append(
            f"  Intrinsic Dimension    "
            f"{fmt(f'{id_.d_int:.1f}', color)}"
            f"   (R²={id_.r_squared:.3f})"
            f"   {fmt(icon + '  ' + id_.interpretation.upper(), color)}"
        )

        # ── Summary line ───────────────────────────────────────────────────────
        lines.append("═" * width)

        if self.n_issues == 0:
            lines.append(fmt("  ✓ No issues detected. Embedding space looks healthy.", "green"))
        elif self.n_issues == 1:
            lines.append(fmt(f"  ⚠ 1 issue detected.", "yellow"))
        else:
            lines.append(fmt(f"  ✗ {self.n_issues} issues detected.", "red"))

        if self.needs_transform:
            lines.append(
                "  → Run audit.transform(embeddings) to fix anisotropy/dimensionality."
            )
        if self.has_brittle_zones:
            lines.append(
                "  → Run audit.get_router() for runtime brittle zone routing."
            )

        lines.append("  → Run report.fix_plan() for detailed recommendations.")
        lines.append("")

        print("\n".join(lines))

    # ── Compare ────────────────────────────────────────────────────────────────

    def compare(self, use_color: bool = True) -> None:
        """
        Show before/after comparison if this report was generated after a transform.

        Only available after calling audit.transform() and re-running audit.run().
        Prints a side-by-side comparison of all five metrics.

        Example
        -------
        >>> fixed = audit.transform(embeddings, strategy='whiten')
        >>> report_after = audit.run(fixed)
        >>> report_after.compare()
        """
        if self._pre_transform_report is None:
            print(
                "No pre-transform report available. "
                "Call audit.transform() first, then audit.run() on the result."
            )
            return

        def fmt(text: str, color: str) -> str:
            return _colorize(text, color) if use_color else text

        before = self._pre_transform_report
        after = self
        width = 58

        lines = [
            "",
            fmt("Spectralyte — Before / After Comparison", "bold"),
            "═" * width,
            f"  {'Metric':<26} {'Before':>10}  {'After':>10}  {'Change':>10}",
            "─" * width,
        ]

        def diff_line(label, before_val, after_val, higher_is_better=True):
            delta = after_val - before_val
            pct = (delta / before_val * 100) if before_val != 0 else 0
            improved = (delta > 0) == higher_is_better
            color = "green" if improved else "red"
            sign = "+" if delta >= 0 else ""
            return (
                f"  {label:<26} {before_val:>10.3f}  "
                f"{fmt(f'{after_val:.3f}', color):>10}  "
                f"{fmt(f'{sign}{pct:.1f}%', color):>10}"
            )

        lines.append(diff_line(
            "Anisotropy Score",
            before.anisotropy.score,
            after.anisotropy.score,
            higher_is_better=False
        ))
        lines.append(diff_line(
            "Utilization",
            before.dimensionality.utilization,
            after.dimensionality.utilization,
            higher_is_better=True
        ))
        lines.append(diff_line(
            "Density CV",
            before.density.cv,
            after.density.cv,
            higher_is_better=False
        ))
        lines.append(diff_line(
            "Retrieval Stability",
            before.sensitivity.mean_stability,
            after.sensitivity.mean_stability,
            higher_is_better=True
        ))
        lines.append(diff_line(
            "Intrinsic Dimension",
            before.intrinsic_dim.d_int,
            after.intrinsic_dim.d_int,
            higher_is_better=True
        ))
        lines.append("═" * width)
        lines.append("")

        print("\n".join(lines))

    # ── Fix plan ───────────────────────────────────────────────────────────────

    def fix_plan(
        self,
        framework: Literal["langchain", "llamaindex", "generic"] = "generic"
    ) -> str:
        """
        Generate a framework-specific remediation plan with working code.

        For each detected issue, provides a specific actionable fix with
        copy-paste code targeting the specified framework. Transforms are
        recommended for anisotropy and dimensionality. Framework-specific
        retrieval strategy changes are recommended for density and sensitivity.

        Parameters
        ----------
        framework : str
            Target framework for generated code. One of 'langchain',
            'llamaindex', or 'generic'. Default 'generic'.

        Returns
        -------
        str
            Formatted remediation plan with actionable recommendations
            and copy-paste code snippets.

        Example
        -------
        >>> plan = report.fix_plan(framework='langchain')
        >>> print(plan)
        """
        lines = [
            "",
            "═" * 56,
            "  Spectralyte — Remediation Plan",
            f"  Framework: {framework}",
            "═" * 56,
            "",
        ]

        has_issues = False

        # ── Anisotropy fix ─────────────────────────────────────────────────────
        if self.anisotropy.interpretation not in {"healthy"}:
            has_issues = True
            lines += [
                f"Issue 1: High Anisotropy (score={self.anisotropy.score:.3f})",
                "─" * 40,
                "Root cause: Embedding vectors cluster along a few directions.",
                "Cosine similarity loses discriminative power.",
                "Fix: Apply whitening or ABTT transform to your embeddings.",
                "",
                "  # Fix anisotropy — no re-embedding required",
                "  from spectralyte import Spectralyte",
                "  audit = Spectralyte(embeddings)",
                "  fixed_embeddings = audit.transform(embeddings, strategy='whiten')",
                "  # Re-index fixed_embeddings in your vector database",
                "",
            ]

        # ── Dimensionality fix ─────────────────────────────────────────────────
        if self.dimensionality.interpretation not in {"healthy", "moderate"}:
            has_issues = True
            lines += [
                f"Issue 2: Low Effective Dimensionality "
                f"({self.dimensionality.effective_dims}/{self.dimensionality.nominal_dims} dims used)",
                "─" * 40,
                "Root cause: Most embedding dimensions carry noise, not signal.",
                f"Fix: Reduce to {self.dimensionality.effective_dims} dimensions via PCA.",
                "Benefits: Faster retrieval, reduced storage, less noise.",
                "",
                "  # Reduce dimensionality — no re-embedding required",
                "  fixed_embeddings = audit.transform(embeddings, strategy='pca_reduce')",
                "  # Re-index fixed_embeddings in your vector database",
                "",
            ]

        # ── Density fix ────────────────────────────────────────────────────────
        if self.density.interpretation not in {"uniform", "moderate"}:
            has_issues = True
            if framework == "langchain":
                code = [
                    "  from langchain.vectorstores import Chroma",
                    "  retriever = vectorstore.as_retriever(",
                    "      search_type='mmr',",
                    "      search_kwargs={'k': 6, 'fetch_k': 20, 'lambda_mult': 0.5}",
                    "  )",
                ]
            elif framework == "llamaindex":
                code = [
                    "  from llama_index.core.retrievers import VectorIndexRetriever",
                    "  retriever = VectorIndexRetriever(",
                    "      index=index,",
                    "      similarity_top_k=6,",
                    "      vector_store_query_mode='mmr',",
                    "  )",
                ]
            else:
                code = [
                    "  # Switch from pure cosine similarity to MMR retrieval",
                    "  # MMR balances relevance with result diversity",
                    "  # Consult your vector DB docs for MMR configuration",
                ]

            lines += [
                f"Issue 3: High Density Clustering (CV={self.density.cv:.3f})",
                "─" * 40,
                f"Root cause: {self.density.n_outliers} outlier embeddings detected.",
                "Queries near cluster boundaries return inconsistent results.",
                "Fix: Switch to Maximum Marginal Relevance (MMR) retrieval.",
                "",
            ] + code + [""]

        # ── Sensitivity fix ────────────────────────────────────────────────────
        if self.sensitivity.interpretation not in {"stable", "moderate"}:
            has_issues = True
            lines += [
                f"Issue 4: High Retrieval Sensitivity "
                f"(stability={self.sensitivity.mean_stability:.3f})",
                "─" * 40,
                f"Root cause: {self.sensitivity.n_brittle} embeddings in brittle zones.",
                "Small query changes produce large result set changes.",
                "Fix: Install the Spectralyte router for intelligent query routing.",
                "",
                "  # Build router from audit results",
                "  router = audit.get_router()",
                "  router.save('spectralyte_router.pkl')",
                "",
                "  # At query time",
                "  from spectralyte import Router",
                "  router = Router.load('spectralyte_router.pkl')",
                "  zone = router.classify(query_embedding)",
                "  # Zone is 'stable', 'brittle', or 'dense_boundary'",
                "",
            ]

        if not has_issues:
            lines += [
                "  No critical issues detected.",
                "  Your embedding space is geometrically healthy.",
                "  No remediation needed at this time.",
            ]

        lines += ["═" * 56, ""]
        return "\n".join(lines)

    # ── Export ─────────────────────────────────────────────────────────────────

    def export(self, path: str) -> None:
        """
        Export structured audit results to a JSON file.

        Parameters
        ----------
        path : str
            File path for the JSON output. Should end in '.json'.

        Example
        -------
        >>> report.export('audit_results.json')
        """
        data = {
            "embeddings_shape": list(self.embeddings_shape),
            "n_issues": self.n_issues,
            "needs_transform": self.needs_transform,
            "has_brittle_zones": self.has_brittle_zones,
            "anisotropy": {
                "score": self.anisotropy.score,
                "interpretation": self.anisotropy.interpretation,
                "n_vectors": self.anisotropy.n_vectors,
                "n_dims": self.anisotropy.n_dims,
                "sampled": self.anisotropy.sampled,
            },
            "dimensionality": {
                "effective_dims": self.dimensionality.effective_dims,
                "nominal_dims": self.dimensionality.nominal_dims,
                "utilization": self.dimensionality.utilization,
                "participation_ratio": self.dimensionality.participation_ratio,
                "variance_threshold": self.dimensionality.variance_threshold,
                "interpretation": self.dimensionality.interpretation,
            },
            "density": {
                "cv": self.density.cv,
                "mean_knn_distance": self.density.mean_knn_distance,
                "std_knn_distance": self.density.std_knn_distance,
                "n_outliers": self.density.n_outliers,
                "interpretation": self.density.interpretation,
                "k": self.density.k,
            },
            "sensitivity": {
                "mean_stability": self.sensitivity.mean_stability,
                "n_brittle": self.sensitivity.n_brittle,
                "brittle_fraction": self.sensitivity.brittle_fraction,
                "epsilon_used": self.sensitivity.epsilon_used,
                "interpretation": self.sensitivity.interpretation,
                "k": self.sensitivity.k,
                "m": self.sensitivity.m,
            },
            "intrinsic_dim": {
                "d_int": self.intrinsic_dim.d_int,
                "r_squared": self.intrinsic_dim.r_squared,
                "interpretation": self.intrinsic_dim.interpretation,
                "trim_fraction": self.intrinsic_dim.trim_fraction,
                "n_points_used": self.intrinsic_dim.n_points_used,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Audit results exported to {path}")

# ── Plot ───────────────────────────────────────────────────────────────────

    def plot(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        show: bool = True,
        save_dir: Optional[str] = None,
    ) -> dict:
        """
        Visualize all five audit metrics.

        Parameters
        ----------
        backend : str
            'matplotlib' for static plots — default, library use, works in
            notebooks and scripts with no extra dependencies.
            'plotly' for interactive plots — web/desktop app use, produces
            HTML with hover tooltips and zoom.
        show : bool
            If True, display plots immediately. Default True.
            Set to False when saving without displaying.
        save_dir : Optional[str]
            If provided, save all figures to this directory.
            Matplotlib saves as .png, Plotly saves as .html.

        Returns
        -------
        dict
            Dictionary mapping metric name to figure object.
            Keys: 'summary', 'anisotropy', 'dimensionality',
                  'density', 'sensitivity', 'intrinsic_dim'

        Example
        -------
        >>> report.plot()
        >>> report.plot(backend='plotly', save_dir='./audit_plots')
        """
        if backend == "matplotlib":
            from spectralyte.visualization.plots_matplotlib import plot_all
        elif backend == "plotly":
            from spectralyte.visualization.plots_plotly import plot_all
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose 'matplotlib' or 'plotly'."
            )
        return plot_all(self, show=show, save_dir=save_dir)