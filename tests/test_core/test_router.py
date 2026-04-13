"""
tests/test_core/test_router.py
================================
Tests for the Router — runtime query zone classification.

Key properties verified:
- Queries near brittle zone centroids classified as 'brittle'
- Queries near boundary zone centroids classified as 'dense_boundary'
- Queries far from all centroids classified as 'stable'
- Stable is the default when no zones detected
- Classification is consistent with centroid distances
- Router serializes and deserializes correctly
- Batch classification matches individual classification
- Input validation raises appropriate errors
- from_report() builds correctly from a real AuditReport
"""

import numpy as np
import pytest
import tempfile
import os
import pickle

from spectralyte.core.router import (
    Router, CentroidSet, Zone,
    _l2_normalize, _l2_normalize_single,
    _compute_centroids_and_radius, _min_cosine_distance
)
from spectralyte import Spectralyte


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def embedding_dim():
    return 64


@pytest.fixture
def stable_embeddings(embedding_dim):
    """Well-separated clusters — produces clear zone structure."""
    rng = np.random.RandomState(42)
    centers = rng.randn(5, embedding_dim) * 10
    clusters = []
    for center in centers:
        cluster = center + rng.randn(40, embedding_dim) * 0.1
        clusters.append(cluster)
    return np.vstack(clusters)


@pytest.fixture
def audit_and_router(stable_embeddings):
    """Full audit and router built from stable embeddings."""
    audit = Spectralyte(stable_embeddings, k=5, random_seed=42)
    report = audit.run(verbose=False)
    router = audit.get_router()
    return audit, report, router, stable_embeddings


@pytest.fixture
def simple_router(embedding_dim):
    """
    Simple router with manually constructed brittle and boundary centroids.
    Makes it easy to write deterministic tests.
    """
    rng = np.random.RandomState(42)

    # Brittle centroid in one direction
    brittle_center = np.zeros(embedding_dim)
    brittle_center[0] = 1.0   # points in +x direction

    brittle_cs = CentroidSet(
        centroids=brittle_center.reshape(1, -1),
        radius=0.2,
        zone="brittle",
        n_source_embeddings=20,
    )

    # Boundary centroid in another direction
    boundary_center = np.zeros(embedding_dim)
    boundary_center[1] = 1.0   # points in +y direction

    boundary_cs = CentroidSet(
        centroids=boundary_center.reshape(1, -1),
        radius=0.2,
        zone="dense_boundary",
        n_source_embeddings=15,
    )

    return Router(
        brittle_centroids=brittle_cs,
        boundary_centroids=boundary_cs,
        embedding_dim=embedding_dim,
        n_index_embeddings=500,
    )


# ── Classification correctness tests ──────────────────────────────────────────

def test_query_near_brittle_centroid_is_brittle(simple_router, embedding_dim):
    """Query close to a brittle centroid should be classified as brittle."""
    # Query pointing in +x direction (near brittle centroid)
    query = np.zeros(embedding_dim)
    query[0] = 1.0
    query += np.random.RandomState(0).randn(embedding_dim) * 0.05
    query = query / np.linalg.norm(query)

    zone = simple_router.classify(query)
    assert zone == "brittle", f"Expected 'brittle', got '{zone}'"


def test_query_near_boundary_centroid_is_boundary(simple_router, embedding_dim):
    """Query close to a boundary centroid should be classified as dense_boundary."""
    # Query pointing in +y direction (near boundary centroid)
    query = np.zeros(embedding_dim)
    query[1] = 1.0
    query += np.random.RandomState(0).randn(embedding_dim) * 0.05
    query = query / np.linalg.norm(query)

    zone = simple_router.classify(query)
    assert zone == "dense_boundary", f"Expected 'dense_boundary', got '{zone}'"


def test_query_far_from_all_centroids_is_stable(simple_router, embedding_dim):
    """Query far from all centroids should be classified as stable."""
    # Query pointing in +z direction (far from both centroids)
    query = np.zeros(embedding_dim)
    query[2] = 1.0   # neither +x nor +y

    zone = simple_router.classify(query)
    assert zone == "stable", f"Expected 'stable', got '{zone}'"


def test_no_zones_always_stable(embedding_dim):
    """Router with no zone centroids should always return stable."""
    router = Router(
        brittle_centroids=None,
        boundary_centroids=None,
        embedding_dim=embedding_dim,
        n_index_embeddings=100,
    )
    rng = np.random.RandomState(42)
    for _ in range(10):
        query = rng.randn(embedding_dim)
        assert router.classify(query) == "stable"


def test_brittle_takes_priority_over_boundary(embedding_dim):
    """When query is in both brittle and boundary zones, brittle wins."""
    # Both centroids pointing in same direction
    center = np.zeros(embedding_dim)
    center[0] = 1.0

    brittle_cs = CentroidSet(
        centroids=center.reshape(1, -1),
        radius=0.5,   # large radius — covers everything
        zone="brittle",
        n_source_embeddings=10,
    )
    boundary_cs = CentroidSet(
        centroids=center.reshape(1, -1),
        radius=0.5,
        zone="dense_boundary",
        n_source_embeddings=10,
    )

    router = Router(
        brittle_centroids=brittle_cs,
        boundary_centroids=boundary_cs,
        embedding_dim=embedding_dim,
        n_index_embeddings=100,
    )

    query = center.copy()
    zone = router.classify(query)
    assert zone == "brittle", "Brittle should take priority over dense_boundary"


def test_only_brittle_no_boundary(embedding_dim):
    """Router with only brittle centroids works correctly."""
    center = np.zeros(embedding_dim)
    center[0] = 1.0

    brittle_cs = CentroidSet(
        centroids=center.reshape(1, -1),
        radius=0.3,
        zone="brittle",
        n_source_embeddings=10,
    )

    router = Router(
        brittle_centroids=brittle_cs,
        boundary_centroids=None,
        embedding_dim=embedding_dim,
        n_index_embeddings=100,
    )

    # Near centroid → brittle
    assert router.classify(center) == "brittle"

    # Far from centroid → stable (no boundary zone)
    far_query = np.zeros(embedding_dim)
    far_query[1] = 1.0
    assert router.classify(far_query) == "stable"


def test_only_boundary_no_brittle(embedding_dim):
    """Router with only boundary centroids works correctly."""
    center = np.zeros(embedding_dim)
    center[0] = 1.0

    boundary_cs = CentroidSet(
        centroids=center.reshape(1, -1),
        radius=0.3,
        zone="dense_boundary",
        n_source_embeddings=10,
    )

    router = Router(
        brittle_centroids=None,
        boundary_centroids=boundary_cs,
        embedding_dim=embedding_dim,
        n_index_embeddings=100,
    )

    assert router.classify(center) == "dense_boundary"

    far_query = np.zeros(embedding_dim)
    far_query[1] = 1.0
    assert router.classify(far_query) == "stable"


# ── Return type tests ──────────────────────────────────────────────────────────

def test_classify_returns_valid_zone(simple_router, embedding_dim):
    """classify() must return one of the three valid zone strings."""
    rng = np.random.RandomState(42)
    for _ in range(20):
        query = rng.randn(embedding_dim)
        zone = simple_router.classify(query)
        assert zone in {"stable", "brittle", "dense_boundary"}


def test_classify_accepts_unnormalized_query(simple_router, embedding_dim):
    """classify() should work on unnormalized input."""
    query = np.zeros(embedding_dim)
    query[0] = 100.0   # large magnitude, same direction as brittle centroid
    zone = simple_router.classify(query)
    assert zone == "brittle"


def test_classify_accepts_2d_input(simple_router, embedding_dim):
    """classify() should accept (1, d) shaped input."""
    query = np.zeros((1, embedding_dim))
    query[0, 0] = 1.0
    zone = simple_router.classify(query)
    assert zone in {"stable", "brittle", "dense_boundary"}


# ── Batch classification tests ─────────────────────────────────────────────────

def test_batch_matches_individual(simple_router, embedding_dim):
    """classify_batch() results must match classify() for each query."""
    rng = np.random.RandomState(42)
    queries = rng.randn(20, embedding_dim)

    batch_zones = simple_router.classify_batch(queries)
    individual_zones = [simple_router.classify(queries[i]) for i in range(len(queries))]

    assert batch_zones == individual_zones


def test_batch_returns_list(simple_router, embedding_dim):
    """classify_batch() must return a list."""
    rng = np.random.RandomState(42)
    queries = rng.randn(10, embedding_dim)
    result = simple_router.classify_batch(queries)
    assert isinstance(result, list)
    assert len(result) == 10


# ── Serialization tests ────────────────────────────────────────────────────────

def test_save_and_load_produces_same_classifications(simple_router, embedding_dim):
    """Saved and loaded router must produce identical classifications."""
    rng = np.random.RandomState(42)
    queries = rng.randn(20, embedding_dim)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        simple_router.save(path)
        loaded = Router.load(path)

        for i in range(len(queries)):
            original_zone = simple_router.classify(queries[i])
            loaded_zone = loaded.classify(queries[i])
            assert original_zone == loaded_zone, (
                f"Query {i}: original={original_zone}, loaded={loaded_zone}"
            )
    finally:
        os.unlink(path)


def test_load_returns_router_instance(simple_router):
    """Router.load() must return a Router instance."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        simple_router.save(path)
        loaded = Router.load(path)
        assert isinstance(loaded, Router)
    finally:
        os.unlink(path)


def test_saved_file_exists(simple_router):
    """save() must create a file at the specified path."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    os.unlink(path)   # delete so we can verify save creates it
    try:
        simple_router.save(path)
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ── from_report() tests ────────────────────────────────────────────────────────

def test_from_report_returns_router(audit_and_router):
    """from_report() must return a Router instance."""
    _, _, router, _ = audit_and_router
    assert isinstance(router, Router)


def test_from_report_correct_dimensions(audit_and_router):
    """Router embedding_dim must match the index."""
    _, _, router, embeddings = audit_and_router
    assert router.embedding_dim == embeddings.shape[1]


def test_from_report_correct_n_embeddings(audit_and_router):
    """Router n_index_embeddings must match the index size."""
    _, _, router, embeddings = audit_and_router
    assert router.n_index_embeddings == len(embeddings)


def test_from_report_router_classifies_all_queries(audit_and_router):
    """Router built from real audit must classify all queries without error."""
    _, _, router, embeddings = audit_and_router
    rng = np.random.RandomState(42)
    queries = rng.randn(20, embeddings.shape[1])
    for i in range(len(queries)):
        zone = router.classify(queries[i])
        assert zone in {"stable", "brittle", "dense_boundary"}


# ── Summary tests ──────────────────────────────────────────────────────────────

def test_summary_returns_string(simple_router):
    """summary() must return a non-empty string."""
    s = simple_router.summary()
    assert isinstance(s, str)
    assert len(s) > 0


def test_summary_contains_zone_info(simple_router):
    """summary() must mention brittle and boundary zones."""
    s = simple_router.summary()
    assert "Brittle" in s
    assert "Boundary" in s or "boundary" in s


# ── Input validation tests ─────────────────────────────────────────────────────

def test_raises_for_wrong_dimension(simple_router, embedding_dim):
    """Wrong query dimension should raise ValueError."""
    wrong_dim_query = np.random.randn(embedding_dim + 10)
    with pytest.raises(ValueError, match="dimensionality"):
        simple_router.classify(wrong_dim_query)


def test_raises_for_3d_input(simple_router, embedding_dim):
    """3D input should raise ValueError."""
    query = np.random.randn(1, 1, embedding_dim)
    with pytest.raises(ValueError):
        simple_router.classify(query)


def test_raises_batch_for_1d_input(simple_router, embedding_dim):
    """1D input to classify_batch() should raise ValueError."""
    query = np.random.randn(embedding_dim)
    with pytest.raises(ValueError):
        simple_router.classify_batch(query)


# ── Helper function tests ──────────────────────────────────────────────────────

def test_l2_normalize_produces_unit_vectors():
    """_l2_normalize should produce vectors with unit norm."""
    rng = np.random.RandomState(42)
    V = rng.randn(100, 64)
    V_norm = _l2_normalize(V)
    norms = np.linalg.norm(V_norm, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_l2_normalize_single_unit_norm():
    """_l2_normalize_single should produce a unit vector."""
    v = np.array([3.0, 4.0])   # norm = 5
    v_norm = _l2_normalize_single(v)
    assert abs(np.linalg.norm(v_norm) - 1.0) < 1e-10


def test_l2_normalize_zero_vector():
    """Zero vector should not cause division by zero."""
    V = np.zeros((10, 32))
    V_norm = _l2_normalize(V)
    assert not np.any(np.isnan(V_norm))


def test_compute_centroids_shape():
    """_compute_centroids_and_radius should return correct shapes."""
    rng = np.random.RandomState(42)
    embeddings = _l2_normalize(rng.randn(100, 32))
    centroids, radius = _compute_centroids_and_radius(embeddings, n_centroids=5, radius_percentile=75)
    assert centroids.shape == (5, 32)
    assert isinstance(radius, float)
    assert radius > 0


def test_compute_centroids_are_normalized():
    """Centroids returned by _compute_centroids_and_radius should be L2-normalized."""
    rng = np.random.RandomState(42)
    embeddings = _l2_normalize(rng.randn(50, 32))
    centroids, _ = _compute_centroids_and_radius(embeddings, n_centroids=3, radius_percentile=75)
    norms = np.linalg.norm(centroids, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_min_cosine_distance_none_centroid_set():
    """_min_cosine_distance with None centroid set returns (None, None)."""
    query = np.array([1.0, 0.0, 0.0])
    dist, radius = _min_cosine_distance(query, None)
    assert dist is None
    assert radius is None


def test_min_cosine_distance_identical_vectors():
    """Distance from a vector to itself should be near 0."""
    v = np.array([1.0, 0.0, 0.0])
    cs = CentroidSet(centroids=v.reshape(1, -1), radius=0.1, zone="brittle", n_source_embeddings=1)
    dist, _ = _min_cosine_distance(v, cs)
    assert abs(dist) < 1e-10


def test_min_cosine_distance_orthogonal_vectors():
    """Distance between orthogonal unit vectors should be 1.0."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    cs = CentroidSet(centroids=v2.reshape(1, -1), radius=0.1, zone="brittle", n_source_embeddings=1)
    dist, _ = _min_cosine_distance(v1, cs)
    assert abs(dist - 1.0) < 1e-10