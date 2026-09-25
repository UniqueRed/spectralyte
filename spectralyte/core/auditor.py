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
from spectralyte.core import transform as _transform


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
    whiten_rcond : float
        Relative floor for covariance eigenvalues during whitening, as a
        fraction of the largest eigenvalue. Directions below it are damped
        rather than inflated. Default 0.01.

        This matters more than it looks. Whitening rescales every direction
        to equal variance, so on an ill-conditioned space a near-null
        direction gets amplified by the inverse square root of a tiny
        eigenvalue — blowing up noise and destroying retrieval. Lower the
        value for a gentler floor, raise it to damp harder.
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
        whiten_rcond: float = 1e-2,
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
        self.whiten_rcond = whiten_rcond
        self.sample_size = sample_size
        self.random_seed = random_seed

        self._report: Optional[AuditReport] = None

        # The matrix the current report was computed from. transform() fits
        # against this, not against whatever array it is handed, so a query
        # lands in the same space as the indexed corpus.
        self._audited: Optional[np.ndarray] = None

        # Fitted transform, populated lazily by fitted_transform().
        self._fit: Optional[_transform.FittedTransform] = None

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

        # A new audit means a new reference space; discard the fit derived
        # from the previous one rather than silently reusing it.
        self._audited = E
        self._fit = None

        if verbose:
            print()

        return report

    # ── Transform ──────────────────────────────────────────────────────────────

    def transform(
        self,
        embeddings: Optional[np.ndarray] = None,
        strategy: Literal["whiten", "abtt", "pca_reduce"] = "whiten",
        abtt_k: int = 3,
    ) -> np.ndarray:
        """
        Apply a geometric correction transform.

        The transform is *fitted once* on the audited matrix and then applied
        to whatever you pass in. That is what makes it usable at query time:
        a single incoming query runs through the same mapping as the indexed
        corpus, so the two stay in one space.

        Transforms correct anisotropy (whiten, abtt) or reduce dimensionality
        (pca_reduce) directly on existing embeddings — no re-embedding
        required. After transforming your corpus, re-index the result, then
        transform each query with the same call before searching.

        Parameters
        ----------
        embeddings : Optional[np.ndarray]
            Vectors to transform, shape (n, d) or a single (d,) vector.
            Defaults to the audited matrix, so ``audit.transform()`` returns
            the corrected corpus. The width must match the audited matrix.
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
            Transformed embeddings, same row count as the input — and 1D if
            the input was 1D. Width is unchanged except for 'pca_reduce',
            which returns (n, effective_dims).

        Raises
        ------
        RuntimeError
            If run() has not been called before transform().
        ValueError
            If strategy is unknown, or the input width does not match the
            audited matrix.

        Example
        -------
        >>> report = audit.run()
        >>> corpus_fixed = audit.transform(strategy='whiten')
        >>> # ... index corpus_fixed ...
        >>> query_fixed = audit.transform(query_vector, strategy='whiten')
        """
        if self._report is None:
            raise RuntimeError(
                "Call audit.run() before audit.transform(). "
                "The audit results are needed to compute the transform."
            )

        X = self._audited if embeddings is None else embeddings

        return self.fitted_transform().apply(
            X, strategy=strategy, abtt_k=abtt_k
        )

    def fitted_transform(self) -> "_transform.FittedTransform":
        """
        The transform fitted on the audited corpus.

        Fitted lazily and cached, so an audit that never transforms pays no
        decomposition. Persist it with ``.save(path)`` to transform queries
        from another process — the parameters have to outlive this instance
        for a transformed index to stay queryable.

        Raises
        ------
        RuntimeError
            If run() has not been called.

        Example
        -------
        >>> audit.run()
        >>> audit.fitted_transform().save("spectralyte_fit.npz")
        """
        if self._report is None:
            raise RuntimeError(
                "Call audit.run() before requesting the fitted transform."
            )
        if self._fit is None:
            self._fit = _transform.fit(
                self._audited,
                effective_dims=self._report.dimensionality.effective_dims,
                whiten_rcond=self.whiten_rcond,
            )
        return self._fit

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