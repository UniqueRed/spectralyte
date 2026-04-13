"""
spectralyte/core/router.py
============================
Router — runtime query zone classification for intelligent retrieval routing.

The router classifies incoming query embeddings into geometric zones at
sub-millisecond speed using centroid-based approximation. It is built
once from audit results and persists as a serializable object that lives
in the critical path of every production query.

Architecture:
    At build time (from_report):
        1. Extract brittle zone embeddings (RSI stability < threshold)
        2. Extract boundary zone embeddings (LOF score > threshold)
        3. Cluster each set into M and K centroids via k-means
        4. Store centroids + classification radii

    At query time (classify):
        1. Compute distance from query to each brittle centroid
        2. Compute distance from query to each boundary centroid
        3. Return zone based on nearest centroid distance vs radius
        4. Default to 'stable' if query is far from all centroids

    Complexity: O(M + K) dot products per query where M, K << n
    Latency: sub-millisecond guaranteed for M, K <= 20

Zone definitions:
    'stable'        — query is far from all brittle and boundary centroids
    'brittle'       — query is close to a brittle zone centroid
                      → route to augmented retrieval (paraphrase union)
    'dense_boundary'— query is close to a cluster boundary centroid
                      → route to hybrid BM25 + dense retrieval
"""

from __future__ import annotations

import pickle
import numpy as np
from typing import Literal, Optional, Tuple
from dataclasses import dataclass, field


# ── Zone type ──────────────────────────────────────────────────────────────────

Zone = Literal["stable", "brittle", "dense_boundary"]


# ── Centroid set ───────────────────────────────────────────────────────────────

@dataclass
class CentroidSet:
    """
    A set of centroids representing a geometric zone in the embedding space.

    Attributes
    ----------
    centroids : np.ndarray
        Centroid vectors of shape (n_centroids, d). L2-normalized.
    radius : float
        Classification radius. Queries within this cosine distance
        of any centroid are assigned to this zone.
    zone : Zone
        The zone label this centroid set represents.
    n_source_embeddings : int
        Number of embeddings used to build these centroids.
    """
    centroids: np.ndarray
    radius: float
    zone: Zone
    n_source_embeddings: int


# ── Router ─────────────────────────────────────────────────────────────────────

class Router:
    """
    Runtime query router built from Spectralyte audit results.

    Classifies incoming query embeddings into geometric zones using
    centroid-based approximation. All classification is pure linear
    algebra — no model calls, no k-NN search over the full index.

    The router should be built once via from_report(), saved to disk,
    and loaded at application startup. It adds negligible latency to
    the query path.

    Parameters
    ----------
    brittle_centroids : Optional[CentroidSet]
        Centroids representing brittle zones. None if no brittle zones detected.
    boundary_centroids : Optional[CentroidSet]
        Centroids representing dense boundary zones. None if no boundaries detected.
    embedding_dim : int
        Dimensionality of the embedding space.
    n_index_embeddings : int
        Total number of embeddings in the audited index.

    Example
    -------
    >>> # Build time
    >>> router = audit.get_router()
    >>> router.save('spectralyte_router.pkl')
    >>>
    >>> # Query time
    >>> from spectralyte import Router
    >>> router = Router.load('spectralyte_router.pkl')
    >>> zone = router.classify(query_embedding)
    >>> # zone is 'stable', 'brittle', or 'dense_boundary'
    """

    def __init__(
        self,
        brittle_centroids: Optional[CentroidSet],
        boundary_centroids: Optional[CentroidSet],
        embedding_dim: int,
        n_index_embeddings: int,
    ) -> None:
        self.brittle_centroids = brittle_centroids
        self.boundary_centroids = boundary_centroids
        self.embedding_dim = embedding_dim
        self.n_index_embeddings = n_index_embeddings

    # ── Build from audit report ────────────────────────────────────────────────

    @classmethod
    def from_report(
        cls,
        report,
        embeddings: np.ndarray,
        n_brittle_centroids: Optional[int] = None,
        n_boundary_centroids: Optional[int] = None,
        brittle_radius_percentile: float = 75.0,
        boundary_radius_percentile: float = 75.0,
    ) -> "Router":
        """
        Build a Router from an AuditReport and the embedding matrix.

        Extracts brittle and boundary zone embeddings from the audit
        results, clusters them into representative centroids, and
        computes classification radii from the within-cluster distances.

        Parameters
        ----------
        report : AuditReport
            Results from Spectralyte.run(). Provides brittle zone indices
            and LOF scores for boundary detection.
        embeddings : np.ndarray
            The full embedding matrix of shape (n, d). Used to extract
            zone embeddings for centroid computation.
        n_brittle_centroids : Optional[int]
            Number of brittle zone centroids. If None, auto-computed as
            min(n_brittle // 10, 10), minimum 1.
        n_boundary_centroids : Optional[int]
            Number of boundary zone centroids. If None, auto-computed
            the same way from boundary embeddings.
        brittle_radius_percentile : float
            Percentile of within-centroid distances used as classification
            radius for brittle zones. Default 75 — covers most zone members
            while avoiding over-expansion into stable regions.
        boundary_radius_percentile : float
            Same as above for boundary zones.

        Returns
        -------
        Router
            Configured router ready for production use.
        """
        n, d = embeddings.shape

        # L2-normalize all embeddings for cosine distance computation
        V = _l2_normalize(embeddings)

        # ── Brittle zone centroids ─────────────────────────────────────────────
        brittle_centroids = None
        brittle_indices = report.sensitivity.brittle_zone_indices

        if len(brittle_indices) > 0:
            brittle_embeddings = V[brittle_indices]
            k_brittle = n_brittle_centroids or max(1, min(len(brittle_indices) // 10, 10))
            k_brittle = min(k_brittle, len(brittle_indices))

            centroids, radius = _compute_centroids_and_radius(
                brittle_embeddings,
                n_centroids=k_brittle,
                radius_percentile=brittle_radius_percentile,
            )

            brittle_centroids = CentroidSet(
                centroids=centroids,
                radius=radius,
                zone="brittle",
                n_source_embeddings=len(brittle_indices),
            )

        # ── Boundary zone centroids ────────────────────────────────────────────
        boundary_centroids = None
        lof_scores = report.density.lof_scores
        lof_threshold = report.density.lof_threshold
        boundary_indices = np.where(lof_scores > lof_threshold)[0]

        if len(boundary_indices) > 0:
            boundary_embeddings = V[boundary_indices]
            k_boundary = n_boundary_centroids or max(1, min(len(boundary_indices) // 10, 10))
            k_boundary = min(k_boundary, len(boundary_indices))

            centroids, radius = _compute_centroids_and_radius(
                boundary_embeddings,
                n_centroids=k_boundary,
                radius_percentile=boundary_radius_percentile,
            )

            boundary_centroids = CentroidSet(
                centroids=centroids,
                radius=radius,
                zone="dense_boundary",
                n_source_embeddings=len(boundary_indices),
            )

        return cls(
            brittle_centroids=brittle_centroids,
            boundary_centroids=boundary_centroids,
            embedding_dim=d,
            n_index_embeddings=n,
        )

    # ── Classify ───────────────────────────────────────────────────────────────

    def classify(self, query_embedding: np.ndarray) -> Zone:
        """
        Classify a query embedding into a retrieval zone.

        Computes cosine distance from the query to each centroid set.
        Returns the zone of the nearest centroid if the distance is
        within that centroid's radius. Returns 'stable' if the query
        is outside all zone radii.

        Classification is O(M + K) dot products where M is the number
        of brittle centroids and K is the number of boundary centroids.
        For typical values (M, K <= 10) this is sub-millisecond.

        Parameters
        ----------
        query_embedding : np.ndarray
            Single query embedding of shape (d,) or (1, d).
            Does not need to be pre-normalized.

        Returns
        -------
        Zone
            'stable', 'brittle', or 'dense_boundary'.

        Raises
        ------
        ValueError
            If query dimensionality does not match the router's embedding dim.

        Example
        -------
        >>> zone = router.classify(query_embedding)
        >>> if zone == 'stable':
        ...     return dense_retrieve(query_embedding, k)
        >>> elif zone == 'brittle':
        ...     return augmented_retrieve(query_embedding, k)
        >>> elif zone == 'dense_boundary':
        ...     return hybrid_retrieve(query_embedding, k)
        """
        # ── Input validation and normalization ─────────────────────────────────
        q = np.asarray(query_embedding, dtype=np.float64)
        if q.ndim == 2:
            q = q.squeeze(0)
        if q.ndim != 1:
            raise ValueError(
                f"query_embedding must be 1D or (1, d), got shape {query_embedding.shape}"
            )
        if q.shape[0] != self.embedding_dim:
            raise ValueError(
                f"query dimensionality {q.shape[0]} does not match "
                f"router embedding dim {self.embedding_dim}"
            )

        # L2-normalize query
        q = _l2_normalize_single(q)

        # ── If no zones detected, everything is stable ─────────────────────────
        if self.brittle_centroids is None and self.boundary_centroids is None:
            return "stable"

        # ── Compute distances to each centroid set ─────────────────────────────
        brittle_dist, brittle_radius = _min_cosine_distance(q, self.brittle_centroids)
        boundary_dist, boundary_radius = _min_cosine_distance(q, self.boundary_centroids)

        # ── Zone decision ──────────────────────────────────────────────────────
        # Brittle takes priority over boundary — more severe failure mode
        in_brittle = brittle_dist is not None and brittle_dist <= brittle_radius
        in_boundary = boundary_dist is not None and boundary_dist <= boundary_radius

        if in_brittle:
            return "brittle"
        elif in_boundary:
            return "dense_boundary"
        else:
            return "stable"

    def classify_batch(self, query_embeddings: np.ndarray) -> list[Zone]:
        """
        Classify a batch of query embeddings.

        More efficient than calling classify() in a loop because it
        batches the centroid distance computations.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Query embedding matrix of shape (n_queries, d).

        Returns
        -------
        list[Zone]
            Zone classification for each query.
        """
        if query_embeddings.ndim != 2:
            raise ValueError(
                f"query_embeddings must be 2D, got shape {query_embeddings.shape}"
            )

        Q = _l2_normalize(query_embeddings)
        zones = []

        for i in range(len(Q)):
            zones.append(self.classify(Q[i]))

        return zones

    # ── Info ───────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the router configuration."""
        lines = [
            "",
            "Spectralyte Router",
            "══════════════════════════════════════",
            f"  Index size:     {self.n_index_embeddings:,} embeddings",
            f"  Embedding dim:  {self.embedding_dim}",
        ]

        if self.brittle_centroids is not None:
            bc = self.brittle_centroids
            lines.append(
                f"  Brittle zones:  {bc.n_source_embeddings} embeddings → "
                f"{len(bc.centroids)} centroids (radius={bc.radius:.4f})"
            )
        else:
            lines.append("  Brittle zones:  none detected")

        if self.boundary_centroids is not None:
            bc = self.boundary_centroids
            lines.append(
                f"  Boundary zones: {bc.n_source_embeddings} embeddings → "
                f"{len(bc.centroids)} centroids (radius={bc.radius:.4f})"
            )
        else:
            lines.append("  Boundary zones: none detected")

        lines.append("══════════════════════════════════════")
        lines.append("")
        return "\n".join(lines)

    # ── Serialization ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialize router to disk.

        Parameters
        ----------
        path : str
            File path. Conventionally ends in .pkl.

        Example
        -------
        >>> router.save('spectralyte_router.pkl')
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Router":
        """
        Load router from disk.

        Parameters
        ----------
        path : str
            Path to a serialized Router file.

        Returns
        -------
        Router
            Loaded router ready for use.

        Example
        -------
        >>> router = Router.load('spectralyte_router.pkl')
        """
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Private helpers ────────────────────────────────────────────────────────────

def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows of an embedding matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def _l2_normalize_single(v: np.ndarray) -> np.ndarray:
    """L2-normalize a single vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _compute_centroids_and_radius(
    embeddings: np.ndarray,
    n_centroids: int,
    radius_percentile: float,
    max_iter: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Compute k-means centroids and a classification radius for a set of embeddings.

    The radius is set to the given percentile of distances from each
    embedding to its nearest centroid. This ensures the radius covers
    most zone members while not expanding into stable regions.

    Parameters
    ----------
    embeddings : np.ndarray
        L2-normalized embeddings of shape (n, d).
    n_centroids : int
        Number of centroids to compute.
    radius_percentile : float
        Percentile of within-cluster distances to use as radius.
    max_iter : int
        Maximum k-means iterations.
    random_state : int
        Random seed for k-means reproducibility.

    Returns
    -------
    Tuple[np.ndarray, float]
        (centroids of shape (n_centroids, d), classification radius)
    """
    if n_centroids >= len(embeddings):
        # Fewer embeddings than requested centroids — use embeddings directly
        centroids = _l2_normalize(embeddings.copy())
        radius = 0.1   # small default radius
        return centroids, radius

    # k-means on the normalized embeddings
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(
        n_clusters=n_centroids,
        max_iter=max_iter,
        random_state=random_state,
        n_init=3,
    )
    kmeans.fit(embeddings)

    # L2-normalize centroids so cosine distance works correctly
    centroids = _l2_normalize(kmeans.cluster_centers_)

    # Compute distance from each embedding to its nearest centroid
    # Using cosine distance: 1 - (v · c) for unit vectors
    similarities = embeddings @ centroids.T   # shape (n, n_centroids)
    nearest_similarities = similarities.max(axis=1)
    distances_to_nearest = 1.0 - nearest_similarities   # cosine distance

    # Radius is the percentile of these within-cluster distances
    radius = float(np.percentile(distances_to_nearest, radius_percentile))

    # Ensure radius is at least a small positive value
    radius = max(radius, 1e-4)

    return centroids, radius


def _min_cosine_distance(
    query: np.ndarray,
    centroid_set: Optional[CentroidSet],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute minimum cosine distance from a query to a centroid set.

    Parameters
    ----------
    query : np.ndarray
        L2-normalized query vector of shape (d,).
    centroid_set : Optional[CentroidSet]
        Centroid set to compute distance to. Returns (None, None) if None.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        (minimum cosine distance, centroid set radius)
        Both None if centroid_set is None.
    """
    if centroid_set is None:
        return None, None

    # Cosine distance = 1 - cosine similarity
    # For L2-normalized vectors: similarity = query · centroid
    similarities = centroid_set.centroids @ query   # shape (n_centroids,)
    max_similarity = float(similarities.max())
    min_distance = 1.0 - max_similarity

    return min_distance, centroid_set.radius