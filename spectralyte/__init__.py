"""
Spectralyte — Illuminating the geometry of your embedding space.

Geometric auditing and remediation for RAG pipelines and vector search.

Quick start
-----------
>>> import numpy as np
>>> from spectralyte import Spectralyte
>>>
>>> embeddings = np.load("my_embeddings.npy")   # shape (n, d)
>>> audit = Spectralyte(embeddings)
>>> report = audit.run()
>>> report.summary()
>>>
>>> # Fix detected issues
>>> fixed = audit.transform(embeddings, strategy="whiten")
>>>
>>> # Get runtime router
>>> router = audit.get_router()
"""

from spectralyte.core.auditor import Spectralyte
from spectralyte.core.report import AuditReport
from spectralyte.core.router import Router

__version__ = "0.2.0"
__author__ = "Adhviklal Thoppe"
__all__ = ["Spectralyte", "AuditReport", "Router"]