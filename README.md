# spectralyte

**Illuminating the geometry of your embedding space.**

[![PyPI version](https://badge.fury.io/py/spectralyte.svg)](https://badge.fury.io/py/spectralyte)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/UniqueRed/spectralyte/actions/workflows/ci.yml/badge.svg)](https://github.com/UniqueRed/spectralyte/actions)

Most RAG failures are not model failures — they are geometry failures. If your embedding space is anisotropic, over-clustered, or low-dimensional, no amount of prompt engineering or model upgrades will fix the underlying retrieval problem.

Spectralyte makes the invisible geometry visible.

```python
from spectralyte import Spectralyte

audit = Spectralyte(embeddings)   # numpy array of shape (n, d)
report = audit.run()
report.summary()
```

```
Spectralyte Audit Report
════════════════════════════════════════
  Anisotropy Score       0.61   ⚠ HIGH
  Effective Dimensions   43 / 1536
  Density CV             0.84   ⚠ CLUSTERED
  Retrieval Stability    0.58   ⚠ BRITTLE
  Intrinsic Dimension    31
════════════════════════════════════════
  2 critical issues detected.
  Run report.plot() for full visualization.
  Run audit.transform('whiten') to fix anisotropy.
```

---

## Why Spectralyte

When a RAG pipeline returns wrong results, engineers have three debugging tools: inspect individual queries, run ad-hoc similarity searches, or stare at raw vectors. None of them answer the foundational question: **is my embedding space geometrically healthy?**

Spectralyte answers that question with five geometric metrics derived from linear algebra and manifold theory — all computed locally, with no API calls and no cost beyond the compute already used to generate your embeddings.

| Metric | What It Catches |
|--------|----------------|
| **Anisotropy** | Vectors clustered in one direction — cosine similarity loses discriminative power |
| **Effective Dimensionality** | Space is lower-dimensional than expected — distinct content collapses together |
| **Density Distribution** | Tight clusters with voids — small query changes flip entire result sets |
| **Retrieval Sensitivity** | Unstable regions — rephrasing a query returns completely different documents |
| **Intrinsic Dimensionality** | True manifold complexity — guides dimensionality reduction decisions |

---

## Installation

```bash
pip install spectralyte
```

Requires Python 3.9+. Core dependencies: `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

For large indices (>50k embeddings), install the optional FAISS backend:

```bash
pip install spectralyte[faiss]
```

---

## Quick Start

```python
import numpy as np
from spectralyte import Spectralyte

# Works with any embedding source — OpenAI, sentence-transformers, Cohere, etc.
# Just pass a numpy array of shape (n, d)
embeddings = np.load("my_embeddings.npy")

# Run the full geometric audit
audit = Spectralyte(embeddings)
report = audit.run()

# Human-readable summary
report.summary()

# Visualize all five metrics
report.plot()

# Export structured results
report.export("audit.json")
```

---

## Fixing Detected Issues

For the two most common problems, Spectralyte can fix them directly — no re-embedding required.

```python
# Fix anisotropy via whitening transform
fixed = audit.transform(embeddings, strategy="whiten")

# Fix anisotropy via All-but-the-Top (ABTT)
fixed = audit.transform(embeddings, strategy="abtt", k=3)

# Re-index with fixed embeddings
# your_vector_db.upsert(fixed)
```

---

## Individual Metrics

Each metric is independently importable if you only need one.

```python
from spectralyte.metrics import anisotropy, effective_dimensionality
from spectralyte.metrics import density, sensitivity, intrinsic_dim

# Anisotropy score (0 = isotropic, 1 = fully anisotropic)
result = anisotropy.compute(embeddings)
print(result.score)

# Effective dimensionality via SVD
result = effective_dimensionality.compute(embeddings, variance_threshold=0.95)
print(result.effective_dims, "/", result.nominal_dims)

# Retrieval Sensitivity Index
result = sensitivity.compute(embeddings, k=10, epsilon=0.05, m=5)
print(result.mean_stability)
```

---

## Use Cases

**Pre-deployment audit** — catch geometry problems before your RAG pipeline reaches production.

**Embedding model comparison** — evaluate models on your specific corpus, not just generic benchmarks.

**Chunking strategy validation** — compare geometric health across different chunking approaches.

**Index drift monitoring** — detect when new documents degrade the geometric structure of your index.

**Debugging retrieval failures** — localize whether failures are geometric (brittle zones) or content-based (missing documents).

---

## Configuration

```python
audit = Spectralyte(
    embeddings,
    k=10,                    # nearest neighbors for density and sensitivity
    sensitivity_epsilon=0.05, # perturbation scale as fraction of mean k-NN distance
    sensitivity_m=5,         # perturbations per embedding for RSI
    variance_threshold=0.95, # cumulative variance for effective dimensionality
    sample_size=None,        # subsample for large indices (auto-set for n > 50k)
    random_seed=42           # reproducibility
)
```

---

## The Math

Spectralyte implements five geometric metrics with rigorous mathematical foundations:

- **Anisotropy** — Gram matrix formulation: measures average pairwise cosine similarity across all embedding pairs
- **Effective Dimensionality** — SVD-based: minimum principal components explaining 95% of variance
- **Density Distribution** — k-NN distance coefficient of variation + Local Outlier Factor
- **Retrieval Sensitivity Index** — Jaccard stability of top-k results under Gaussian perturbation
- **Intrinsic Dimensionality** — TwoNN estimator (Facco et al. 2017): fits d_int from nearest-neighbor distance ratios

Full mathematical derivations are available in the [documentation](https://spectralyte.readthedocs.io).

---

## Roadmap

- [x] Anisotropy metric
- [x] Effective dimensionality
- [x] Density distribution
- [x] Retrieval Sensitivity Index
- [x] Intrinsic dimensionality (TwoNN)
- [x] Whitening transform
- [x] All-but-the-Top (ABTT) transform
- [ ] CLI entrypoint (`spectralyte audit embeddings.npy`)
- [ ] HTML report export
- [ ] FAISS backend for large indices
- [ ] Pinecone / Qdrant / Weaviate native integration
- [ ] Continuous drift monitoring

---

## Contributing

Contributions are welcome. Please open an issue before submitting a PR so we can discuss the change.

```bash
git clone https://github.com/UniqueRed/spectralyte
cd spectralyte
pip install -e ".[dev]"
pytest
```

---

## Citation

If you use Spectralyte in research, please cite:

```bibtex
@software{spectralyte2026,
  author = {Thoppe, Adhviklal},
  title = {Spectralyte: Illuminating the Geometry of Your Embedding Space},
  year = {2026},
  url = {https://github.com/UniqueRed/spectralyte}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.