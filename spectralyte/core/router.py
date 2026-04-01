from __future__ import annotations
import numpy as np
import pickle
from typing import Literal


class Router:
    """Runtime query router — stub implementation."""

    def __init__(self, brittle_indices, embeddings):
        self.brittle_indices = brittle_indices
        self._centroids = None
        if len(brittle_indices) > 0 and embeddings is not None:
            self._brittle_centroids = embeddings[brittle_indices].mean(axis=0)
        else:
            self._brittle_centroids = None

    @classmethod
    def from_report(cls, report, embeddings):
        return cls(report.sensitivity.brittle_zone_indices, embeddings)

    def classify(self, query_embedding: np.ndarray) -> Literal["stable", "brittle", "dense_boundary"]:
        if self._brittle_centroids is None:
            return "stable"
        return "stable"

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Router":
        with open(path, 'rb') as f:
            return pickle.load(f)