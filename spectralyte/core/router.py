"""
Router — runtime query classification for intelligent retrieval routing.
"""

from __future__ import annotations
import numpy as np
from typing import Literal
import pickle


class Router:
    """
    Runtime query router built from Spectralyte audit results.

    Classifies incoming query embeddings into geometric zones and
    recommends the appropriate retrieval strategy for each zone.
    All classification is pure linear algebra — sub-millisecond latency.

    Example
    -------
    >>> router = audit.get_router()
    >>> router.save('router.pkl')
    >>> # At runtime:
    >>> router = Router.load('router.pkl')
    >>> zone = router.classify(query_embedding)
    >>> # zone is one of: 'stable', 'brittle', 'dense_boundary'
    """

    def classify(
        self,
        query_embedding: np.ndarray
    ) -> Literal["stable", "brittle", "dense_boundary"]:
        """
        Classify a query embedding into a retrieval zone.

        Parameters
        ----------
        query_embedding : np.ndarray
            Single query embedding of shape (d,).

        Returns
        -------
        str
            Zone classification: 'stable', 'brittle', or 'dense_boundary'.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Serialize router to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Router":
        """Load router from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)