"""
spectralyte/core/auditor.py
=============================
Spectralyte — the primary entry point for geometric auditing.

Orchestrates all five metrics and produces a unified AuditReport.
Also provides transform() for direct embedding correction and
get_router() for runtime query classification.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Literal

from spectralyte.metrics import anisotropy, dimensionality, density, sensitivity, intrinsic_dim
from spectralyte.core.report import AuditReport


class Spectralyte:
    """
    Geometric auditor for embedding spaces.

    The primary entry point for Spectralyte. Accepts an embedding matrix
    and orchestrates all five geometric metrics to produce a unified
    AuditReport. Also provides transform() for direct embedding correction
    and get_router() for runtime query routing.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (n, d). Any embedding model or dimension.
        Does not need to be pre-normalized.
    k : int
        Number of nearest neighbors for density and sensitivity metrics.
        Default 10.
    sensitivity_epsilon : float
        Perturbation scale for RSI as fraction of mean k-NN distance.
        Default 0.05.
    sensitivity_m : int
        Number of perturbations per embedding for RSI. Default 5.
    variance_threshold : float
        Cumulative variance threshold for effective dimensionality.
        Default 0.95 (95%).
    sample_size : Optional[int]
        If set, subsample index for expensive metrics. If None, auto-set
        based on index size. Default None.
    random_seed : int
        Reproducibility seed for all sampling and perturbation. Default 42.

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
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D array of shape (n, d), "
                f"got shape {embeddings.shape}"
            )
        if embeddings.shape[0] < 3:
            raise ValueError(
                f"Spectralyte requires at least 3 embeddings, "
                f"got {embeddings.shape[0]}"
            )

        self.embeddings = embeddings
        self.k = k
        self.sensitivity_epsilon = sensitivity_epsilon
        self.sensitivity_m = sensitivity_m
        self.variance_threshold = variance_threshold
        self.sample_size = sample_size
        self.random_seed = random_seed

        self._report: Optional[AuditReport] = None
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._whitening_matrix: Optional[np.ndarray] = None

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(
        self,
        embeddings: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> AuditReport:
        """
        Run the full geometric audit on the embedding matrix.

        Computes all five metrics in order, sharing intermediate results
        where possible to avoid redundant computation. Returns a unified
        AuditReport with results, interpretation, and remediation guidance.

        Parameters
        ----------
        embeddings : Optional[np.ndarray]
            Embedding matrix to audit. If None, uses the matrix passed to
            __init__. Pass a different matrix here to audit transformed
            embeddings without creating a new Spectralyte instance.
        verbose : bool
            If True, prints progress during computation. Default True.

        Returns
        -------
        AuditReport
            Unified audit results with summary, compare, fix_plan, and
            export methods.

        Example
        -------
        >>> report = audit.run()
        >>> report.summary()
        """
        E = embeddings if embeddings is not None else self.embeddings
        n, d = E.shape

        if verbose:
            print(f"\nSpectralyte — auditing {n:,} × {d} embeddings...\n")

        # ── Metric 1: Anisotropy ───────────────────────────────────────────────
        if verbose:
            print("  [1/5] Computing anisotropy...", end=" ", flush=True)

        anisotropy_result = anisotropy.compute(
            E,
            sample_size=self.sample_size or 5000,
            random_seed=self.random_seed,
        )

        if verbose:
            print(
                f"score={anisotropy_result.score:.3f} "
                f"({anisotropy_result.interpretation})"
            )

        # ── Metric 2: Effective Dimensionality ────────────────────────────────
        if verbose:
            print("  [2/5] Computing effective dimensionality...", end=" ", flush=True)

        dimensionality_result = dimensionality.compute(
            E,
            variance_threshold=self.variance_threshold,
            random_seed=self.random_seed,
        )

        if verbose:
            print(
                f"{dimensionality_result.effective_dims}/{d} dims "
                f"({dimensionality_result.interpretation})"
            )

        # ── Metric 3: Density Distribution ────────────────────────────────────
        if verbose:
            print("  [3/5] Computing density distribution...", end=" ", flush=True)

        density_sample = self.sample_size or min(10_000, n)
        density_result = density.compute(
            E,
            k=self.k,
            sample_size=density_sample,
            random_seed=self.random_seed,
        )

        if verbose:
            print(
                f"CV={density_result.cv:.3f} "
                f"({density_result.interpretation})"
            )

        # ── Metric 4: Retrieval Sensitivity Index ─────────────────────────────
        if verbose:
            print("  [4/5] Computing retrieval sensitivity...", end=" ", flush=True)

        sensitivity_result = sensitivity.compute(
            E,
            k=self.k,
            m=self.sensitivity_m,
            epsilon_fraction=self.sensitivity_epsilon,
            sample_size=self.sample_size,
            random_seed=self.random_seed,
        )

        if verbose:
            print(
                f"stability={sensitivity_result.mean_stability:.3f} "
                f"({sensitivity_result.interpretation})"
            )

        # ── Metric 5: Intrinsic Dimensionality ────────────────────────────────
        if verbose:
            print("  [5/5] Computing intrinsic dimensionality...", end=" ", flush=True)

        intrinsic_result = intrinsic_dim.compute(
            E,
            sample_size=self.sample_size or 5000,
            random_seed=self.random_seed,
        )

        if verbose:
            print(
                f"d_int={intrinsic_result.d_int:.1f} "
                f"(R²={intrinsic_result.r_squared:.3f})"
            )

        # ── Assemble report ───────────────────────────────────────────────────
        report = AuditReport(
            anisotropy=anisotropy_result,
            dimensionality=dimensionality_result,
            density=density_result,
            sensitivity=sensitivity_result,
            intrinsic_dim=intrinsic_result,
            embeddings_shape=(n, d),
            _pre_transform_report=self._report,
        )

        self._report = report

        if verbose:
            print()

        return report

    # ── Transform ──────────────────────────────────────────────────────────────

    def transform(
        self,
        embeddings: np.ndarray,
        strategy: Literal["whiten", "abtt", "pca_reduce"] = "whiten",
        abtt_k: int = 3,
    ) -> np.ndarray:
        """
        Apply a geometric correction transform to an embedding matrix.

        Transforms correct anisotropy (whiten, abtt) or reduce dimensionality
        (pca_reduce) directly on existing embeddings — no re-embedding required.
        After transforming, re-index the result in your vector database.

        Note: Also transform incoming query embeddings at runtime using the
        same strategy before searching, to ensure consistency.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix to transform. Shape (n, d).
        strategy : str
            Transform to apply:
            - 'whiten': isotropic covariance transform (reduces anisotropy)
            - 'abtt': All-but-the-Top, removes dominant directions
            - 'pca_reduce': reduces to effective dimensionality
        abtt_k : int
            Number of top components to remove for ABTT. Default 3.

        Returns
        -------
        np.ndarray
            Transformed embeddings. Same shape as input unless
            strategy='pca_reduce', which returns shape (n, effective_dims).

        Raises
        ------
        RuntimeError
            If run() has not been called before transform().
        ValueError
            If strategy is not one of the three valid options.

        Example
        -------
        >>> report = audit.run()
        >>> fixed = audit.transform(embeddings, strategy='whiten')
        >>> report_after = audit.run(fixed)
        >>> report_after.compare()
        """
        if self._report is None:
            raise RuntimeError(
                "Call audit.run() before audit.transform(). "
                "The audit results are needed to compute the transform."
            )

        if strategy == "whiten":
            return self._whiten(embeddings)
        elif strategy == "abtt":
            return self._abtt(embeddings, k=abtt_k)
        elif strategy == "pca_reduce":
            return self._pca_reduce(embeddings)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose one of: 'whiten', 'abtt', 'pca_reduce'."
            )

    def _whiten(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Whitening transform — makes covariance matrix equal to identity.

        Computes covariance C = (1/n) V^T V, then applies C^(-1/2) via
        eigendecomposition. The result has isotropic covariance structure,
        which corrects anisotropy.

        Transformation: V_white = V @ C^(-1/2)
        Then L2-normalize rows.
        """
        V = embeddings - embeddings.mean(axis=0)
        n = V.shape[0]
        C = (V.T @ V) / n   # covariance matrix

        # Eigendecomposition of symmetric covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # Clip small/negative eigenvalues for numerical stability
        eigenvalues = np.clip(eigenvalues, 1e-10, None)

        # C^(-1/2) = Q @ diag(lambda^(-1/2)) @ Q^T
        whitening = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T

        V_white = V @ whitening

        # L2 normalize
        norms = np.linalg.norm(V_white, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return V_white / norms

    def _abtt(self, embeddings: np.ndarray, k: int = 3) -> np.ndarray:
        """
        All-but-the-Top transform — removes dominant principal components.

        The top k principal components of the embedding matrix tend to
        capture corpus-level bias rather than document-specific semantics.
        Removing them reveals the document-level variation underneath.

        Transformation: V -= V @ U_k @ U_k^T
        Then L2-normalize rows.
        """
        V = embeddings - embeddings.mean(axis=0)

        # Compute top-k left singular vectors via SVD
        # Only need left singular vectors (U)
        U, _, _ = np.linalg.svd(V, full_matrices=False)
        U_k = U[:, :k]   # shape (n, k) — top k left singular vectors

        # Project out the top-k components
        # V -= V @ U_k @ U_k^T projects onto the orthogonal complement
        # Corrected: use right singular vectors for column space projection
        _, _, Vt = np.linalg.svd(V, full_matrices=False)
        W_k = Vt[:k, :].T   # shape (d, k) — top k right singular vectors

        V_abtt = V - V @ W_k @ W_k.T

        # L2 normalize
        norms = np.linalg.norm(V_abtt, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return V_abtt / norms

    def _pca_reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """
        PCA dimensionality reduction to effective dimensionality.

        Reduces embeddings from nominal_dims to effective_dims using the
        principal components identified in the audit. Removes noise dimensions
        while preserving the meaningful variance structure.

        Returns shape (n, effective_dims).
        """
        k = self._report.dimensionality.effective_dims
        V = embeddings - embeddings.mean(axis=0)

        # Use SVD to get principal components
        _, _, Vt = np.linalg.svd(V, full_matrices=False)
        components = Vt[:k, :]   # shape (k, d)

        # Project embeddings onto top-k components
        return V @ components.T   # shape (n, k)

    # ── Router ────────────────────────────────────────────────────────────────

    def get_router(self):
        """
        Build and return a Router from audit results.

        The Router classifies incoming query embeddings into geometric zones
        at runtime (sub-millisecond) and selects the appropriate retrieval
        strategy for each zone.

        Returns
        -------
        Router
            Configured router ready for production use.

        Raises
        ------
        RuntimeError
            If run() has not been called before get_router().

        Example
        -------
        >>> router = audit.get_router()
        >>> router.save('router.pkl')
        >>> # At query time:
        >>> zone = router.classify(query_embedding)
        """
        if self._report is None:
            raise RuntimeError(
                "Call audit.run() before audit.get_router()."
            )

        from spectralyte.core.router import Router
        return Router.from_report(self._report, self.embeddings)