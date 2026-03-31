"""
AuditReport — structured results from a Spectralyte audit.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class AuditReport:
    """
    Structured results from a full Spectralyte audit.

    Attributes
    ----------
    anisotropy_score : float
        Mean pairwise cosine similarity. 0 = isotropic, 1 = fully anisotropic.
    effective_dims : int
        Minimum principal components explaining variance_threshold of variance.
    nominal_dims : int
        Nominal embedding dimensionality.
    density_cv : float
        Coefficient of variation of k-NN distances. High = clustered.
    retrieval_stability : float
        Mean Jaccard stability under perturbation. Low = brittle.
    intrinsic_dim : float
        TwoNN estimate of true manifold dimensionality.
    """

    anisotropy_score: float = 0.0
    effective_dims: int = 0
    nominal_dims: int = 0
    density_cv: float = 0.0
    retrieval_stability: float = 0.0
    intrinsic_dim: float = 0.0
    stability_per_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    lof_scores: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> None:
        """Print a human-readable audit summary."""
        raise NotImplementedError

    def plot(self) -> None:
        """Visualize all five metrics."""
        raise NotImplementedError

    def compare(self) -> None:
        """Show before/after comparison after a transform."""
        raise NotImplementedError

    def fix_plan(
        self,
        framework: Literal["langchain", "llamaindex", "generic"] = "generic"
    ) -> str:
        """
        Generate a framework-specific remediation plan with working code.

        Parameters
        ----------
        framework : str
            Target framework for generated code snippets.

        Returns
        -------
        str
            Formatted remediation plan with copy-paste code.
        """
        raise NotImplementedError

    def export(self, path: str) -> None:
        """Export structured results to JSON."""
        raise NotImplementedError

    @property
    def needs_transform(self) -> bool:
        """True if anisotropy or dimensionality issues detected."""
        raise NotImplementedError