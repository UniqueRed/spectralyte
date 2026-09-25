"""
spectralyte/core/transform.py
=============================
A fitted geometric correction transform, detached from the auditor.

Why this is its own object
--------------------------
A transform is only useful if the *same* mapping reaches both sides of a
search: the corpus you index and every query you later send against it. That
means the fitted parameters have to outlive the :class:`Spectralyte` instance
that produced them — across a process boundary, and in particular across the
CLI, where the audit runs once and queries arrive long afterwards.

Holding the parameters here rather than inside the auditor gives one
implementation of "apply", shared by the library and the CLI, and something
concrete to serialize.

The stored parameters are all derived from the audited corpus:

``mean``
    Column means, subtracted before every projection.
``whitening``
    C^(-1/2), with eigenvalues floored relative to the largest (see
    ``whiten_rcond``).
``right_singular_vectors``
    Right singular vectors of the centered corpus. Serves ABTT for any
    ``abtt_k`` and ``pca_reduce`` for the measured effective dimensionality,
    so neither needs a refit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Strategy = Literal["whiten", "abtt", "pca_reduce"]

STRATEGIES: tuple[str, ...] = ("whiten", "abtt", "pca_reduce")

#: Bumped when the stored field layout changes incompatibly.
FORMAT_VERSION = 1


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows, leaving zero rows untouched."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


@dataclass(frozen=True)
class FittedTransform:
    """
    Transform parameters fitted on an audited corpus.

    Apply with :meth:`apply`. Persist with :meth:`save` and reload with
    :meth:`load` to transform queries in a later process.

    Example
    -------
    >>> audit.run()
    >>> fit = audit.fitted_transform()
    >>> fit.save("spectralyte_fit.npz")
    >>> # ... later, in the serving process ...
    >>> fit = FittedTransform.load("spectralyte_fit.npz")
    >>> query_vector = fit.apply(raw_query, strategy="whiten")
    """

    mean: np.ndarray
    whitening: np.ndarray
    right_singular_vectors: np.ndarray
    effective_dims: int
    whiten_rcond: float

    @property
    def n_dims(self) -> int:
        """Width of the space this transform was fitted on."""
        return int(self.mean.shape[0])

    # ── Applying ───────────────────────────────────────────────────────────────

    def apply(
        self,
        embeddings: np.ndarray,
        strategy: Strategy = "whiten",
        abtt_k: int = 3,
    ) -> np.ndarray:
        """
        Apply the fitted transform.

        Parameters
        ----------
        embeddings : np.ndarray
            Vectors of shape (n, d), or a single (d,) vector. The width must
            match the space this transform was fitted on.
        strategy : str
            'whiten', 'abtt', or 'pca_reduce'.
        abtt_k : int
            Number of dominant directions to remove for ABTT. Default 3.

        Returns
        -------
        np.ndarray
            Transformed vectors, 1D if the input was 1D.
        """
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose one of: {', '.join(repr(s) for s in STRATEGIES)}."
            )

        X = np.asarray(embeddings)

        # A single query arrives as (d,); accept it and answer in kind.
        was_1d = X.ndim == 1
        if was_1d:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(
                f"embeddings must be 1D (d,) or 2D (n, d), got shape {X.shape}"
            )
        if X.shape[1] != self.n_dims:
            raise ValueError(
                f"embeddings have {X.shape[1]} dimensions but this transform was "
                f"fitted on {self.n_dims}-dimensional vectors. A transform fitted "
                f"on one space cannot be applied to another."
            )

        centered = X - self.mean

        if strategy == "whiten":
            out = _l2_normalize(centered @ self.whitening)
        elif strategy == "abtt":
            if abtt_k < 0:
                raise ValueError(f"abtt_k must be non-negative, got {abtt_k}")
            W_k = self.right_singular_vectors[:abtt_k, :].T       # (d, k)
            out = _l2_normalize(centered - centered @ W_k @ W_k.T)
        else:
            components = self.right_singular_vectors[:self.effective_dims, :]
            out = centered @ components.T

        return out[0] if was_1d else out

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Write the fitted parameters to a ``.npz`` file.

        Plain arrays via ``numpy.savez`` — no pickle, so loading a file from
        elsewhere cannot execute code.
        """
        np.savez(
            path,
            format_version=np.array(FORMAT_VERSION),
            mean=self.mean,
            whitening=self.whitening,
            right_singular_vectors=self.right_singular_vectors,
            effective_dims=np.array(self.effective_dims),
            whiten_rcond=np.array(self.whiten_rcond),
        )

    @classmethod
    def load(cls, path: str) -> "FittedTransform":
        """Load fitted parameters written by :meth:`save`."""
        with np.load(path) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"'{path}' uses transform format version {version}, but this "
                    f"version of Spectralyte reads version {FORMAT_VERSION}. "
                    f"Re-run the audit to regenerate it."
                )
            return cls(
                mean=data["mean"],
                whitening=data["whitening"],
                right_singular_vectors=data["right_singular_vectors"],
                effective_dims=int(data["effective_dims"]),
                whiten_rcond=float(data["whiten_rcond"]),
            )


def fit(
    embeddings: np.ndarray,
    effective_dims: int,
    whiten_rcond: float = 1e-2,
) -> FittedTransform:
    """
    Fit the transform parameters on a corpus.

    One centered decomposition serves all three strategies.

    Parameters
    ----------
    embeddings : np.ndarray
        The audited corpus, shape (n, d).
    effective_dims : int
        Target width for ``pca_reduce``, from the dimensionality metric.
    whiten_rcond : float
        Relative floor for covariance eigenvalues, as a fraction of the
        largest. Whitening scales each direction by lambda^(-1/2), so an
        absolute floor lets a near-null direction be amplified without bound,
        drowning the signal in noise. A relative floor caps the gain at
        ``whiten_rcond ** -0.5`` however degenerate the spectrum is.
    """
    mean = embeddings.mean(axis=0)
    V = embeddings - mean

    C = (V.T @ V) / V.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    floor = eigenvalues.max() * whiten_rcond
    eigenvalues = np.clip(eigenvalues, max(floor, 1e-12), None)
    whitening = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T

    _, _, Vt = np.linalg.svd(V, full_matrices=False)

    return FittedTransform(
        mean=mean,
        whitening=whitening,
        right_singular_vectors=Vt,
        effective_dims=int(effective_dims),
        whiten_rcond=float(whiten_rcond),
    )
