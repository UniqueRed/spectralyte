"""
Spectralyte — core auditor.

The Spectralyte class is the primary entry point. It accepts an embedding
matrix of shape (n, d) and orchestrates all five geometric metrics.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from .report import AuditReport


class Spectralyte:
    """
    Geometric auditor for embedding spaces.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (n, d). Any embedding model.
    k : int
        Number of nearest neighbors for density and sensitivity metrics.
    sensitivity_epsilon : float
        Perturbation scale as fraction of mean k-NN distance.
    sensitivity_m : int
        Number of perturbations per embedding for RSI.
    variance_threshold : float
        Cumulative variance threshold for effective dimensionality.
    sample_size : Optional[int]
        If set, subsample index for expensive metrics. Auto-set for n > 50000.
    random_seed : int
        Reproducibility seed.

    Example
    -------
    >>> import numpy as np
    >>> from spectralyte import Spectralyte
    >>> embeddings = np.random.randn(1000, 384)
    >>> audit = Spectralyte(embeddings)
    >>> report = audit.run()
    >>> report.summary()
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        k: int = 10,
        sensitivity_epsilon: float = 0.05,
        sensitivity_m: int = 5,
        variance_threshold: float = 0.95,
        sample_size: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        self.embeddings = embeddings
        self.k = k
        self.sensitivity_epsilon = sensitivity_epsilon
        self.sensitivity_m = sensitivity_m
        self.variance_threshold = variance_threshold
        self.sample_size = sample_size
        self.random_seed = random_seed
        self._report: Optional[AuditReport] = None

    def run(self) -> AuditReport:
        """Run the full geometric audit and return an AuditReport."""
        raise NotImplementedError

    def transform(
        self,
        embeddings: np.ndarray,
        strategy: Literal["whiten", "abtt", "pca_reduce"] = "whiten",
        **kwargs,
    ) -> np.ndarray:
        """
        Apply a geometric correction transform to embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix to transform. Shape (n, d).
        strategy : str
            One of 'whiten', 'abtt', 'pca_reduce'.

        Returns
        -------
        np.ndarray
            Transformed embeddings, same shape unless strategy='pca_reduce'.
        """
        raise NotImplementedError

    def get_router(self):
        """
        Build and return a Router from audit results.

        The Router classifies incoming query embeddings into geometric zones
        at runtime and selects the appropriate retrieval strategy.

        Returns
        -------
        Router
            Configured router ready for production use.
        """
        raise NotImplementedError