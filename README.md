# spectralyte

**Illuminating the geometry of your embedding space.**

[![PyPI version](https://badge.fury.io/py/spectralyte.svg)](https://badge.fury.io/py/spectralyte)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/UniqueRed/spectralyte/actions/workflows/ci.yml/badge.svg)](https://github.com/UniqueRed/spectralyte/actions)
[![Downloads](https://static.pepy.tech/badge/spectralyte)](https://pepy.tech/project/spectralyte)

Most RAG pipeline failures are not model failures — they are geometry failures. If your embedding space is anisotropic, over-clustered, or low-dimensional, no amount of prompt engineering or model upgrades will fix the underlying retrieval problem.

Spectralyte makes the invisible geometry visible — and fixes it.

```python
from spectralyte import Spectralyte

audit = Spectralyte(embeddings)   # numpy array of shape (n, d)
report = audit.run()
report.summary()
```

```
Spectralyte Audit Report
════════════════════════════════════════════════════
  Embeddings: 500 vectors × 384 dims
────────────────────────────────────────────────────
  Anisotropy Score       0.062   ✓  HEALTHY
  Effective Dimensions   183 / 384  (47.7%)   ✓  HEALTHY
  Density CV             0.058   ✓  UNIFORM
  Retrieval Stability    0.953   ✓  STABLE
  Intrinsic Dimension    8.2   (R²=0.888)   ✓  LOW
════════════════════════════════════════════════════
  ✓ No issues detected. Embedding space looks healthy.
```

---

## Why Spectralyte

When a RAG pipeline returns wrong results, engineers have no systematic tool to diagnose why. They can inspect individual queries, run ad-hoc similarity searches, or stare at raw vectors — but nothing answers the foundational question: **is my embedding space geometrically healthy?**

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

For interactive Plotly visualizations:

```bash
pip install spectralyte plotly
```

---

## Quick Start

```python
import numpy as np
from spectralyte import Spectralyte

# Works with any embedding source — OpenAI, sentence-transformers, Cohere, etc.
embeddings = np.load("my_embeddings.npy")   # shape (n, d)

# Run the full geometric audit
audit = Spectralyte(embeddings)
report = audit.run()

# Human-readable summary
report.summary()

# Visualize all five metrics (matplotlib by default)
report.plot()

# Interactive Plotly visualization
report.plot(backend="plotly")

# Export structured results
report.export("audit.json")
```

---

## Fixing Detected Issues

For the most common problems, Spectralyte corrects them directly — no re-embedding required.

```python
# Fix anisotropy via whitening transform
fixed = audit.transform(embeddings, strategy="whiten")

# Fix anisotropy via All-but-the-Top (ABTT)
fixed = audit.transform(embeddings, strategy="abtt", abtt_k=3)

# Reduce to effective dimensionality (saves storage, speeds retrieval)
fixed = audit.transform(embeddings, strategy="pca_reduce")

# Re-audit to verify improvement
audit_fixed = Spectralyte(fixed)
report_fixed = audit_fixed.run()
report_fixed._pre_transform_report = report
report_fixed.compare()
```

```
Spectralyte — Before / After Comparison
══════════════════════════════════════════════════════════
  Metric                     Before       After       Change
──────────────────────────────────────────────────────────
  Anisotropy Score            0.610  →    0.089      -85.4%
  Utilization                 2.8%   →    2.8%        —
  Retrieval Stability         0.580  →    0.810      +39.7%
══════════════════════════════════════════════════════════
```

---

## Runtime Router

Build a router from audit results that classifies every incoming query into a geometric zone and selects the optimal retrieval strategy — in sub-millisecond time.

```python
# Build time: generate router from audit
router = audit.get_router()
router.save("spectralyte_router.pkl")

# Query time: intelligent routing
from spectralyte import Router
router = Router.load("spectralyte_router.pkl")

def retrieve(query_embedding, k=10):
    zone = router.classify(query_embedding)

    if zone == "stable":
        return dense_retrieve(query_embedding, k)
    elif zone == "brittle":
        return augmented_retrieve(query_embedding, k)     # paraphrase union
    elif zone == "dense_boundary":
        return hybrid_retrieve(query_embedding, k)        # BM25 + dense
```

Router classification is pure linear algebra — O(M + K) dot products where M and K are the number of zone centroids. Adds negligible latency to the query path.

---

## Remediation Plan

For issues that can't be fixed by transforming embeddings, Spectralyte generates framework-specific remediation code.

```python
plan = report.fix_plan(framework="langchain")
print(plan)
```

```
════════════════════════════════════════════════════════
  Spectralyte — Remediation Plan
  Framework: langchain
════════════════════════════════════════════════════════

Issue 3: High Density Clustering (CV=0.84)
────────────────────────────────────────
Root cause: 847 boundary documents detected.
Fix: Switch to Maximum Marginal Relevance (MMR) retrieval.

  from langchain.vectorstores import Chroma
  retriever = vectorstore.as_retriever(
      search_type="mmr",
      search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5}
  )
```

---

## Individual Metrics

Each metric is independently importable.

```python
from spectralyte.metrics import anisotropy, dimensionality
from spectralyte.metrics import density, sensitivity, intrinsic_dim

# Anisotropy score (0 = isotropic, 1 = fully anisotropic)
result = anisotropy.compute(embeddings)
print(result.score, result.interpretation)

# Effective dimensionality via SVD
result = dimensionality.compute(embeddings, variance_threshold=0.95)
print(f"{result.effective_dims} / {result.nominal_dims} dims used")

# Density distribution
result = density.compute(embeddings, k=10)
print(f"CV={result.cv:.3f}  ({result.n_outliers} outliers)")

# Retrieval Sensitivity Index
result = sensitivity.compute(embeddings, k=10, epsilon_fraction=0.05)
print(f"Mean stability: {result.mean_stability:.3f}")

# TwoNN intrinsic dimensionality
result = intrinsic_dim.compute(embeddings)
print(f"d_int={result.d_int:.1f}  R²={result.r_squared:.3f}")
```

---

## Configuration

```python
audit = Spectralyte(
    embeddings,
    k=10,                     # nearest neighbors for density and sensitivity
    sensitivity_epsilon=0.05, # perturbation scale as fraction of mean k-NN distance
    sensitivity_m=5,          # perturbations per embedding for RSI
    variance_threshold=0.95,  # cumulative variance for effective dimensionality
    sample_size=None,         # subsample for large indices (auto-set for n > 50k)
    random_seed=42,           # reproducibility
)
```

---

## Use Cases

**Pre-deployment audit** — catch geometry problems before your RAG pipeline reaches production.

**Embedding model comparison** — evaluate models on your specific corpus, not just generic benchmarks.

**Chunking strategy validation** — compare geometric health across different chunking approaches.

**Debugging retrieval failures** — localize whether failures are geometric or content-based.

---

## The Mathematics

Spectralyte implements five metrics with rigorous mathematical foundations:

- **Anisotropy** — Gram matrix: mean off-diagonal cosine similarity across all embedding pairs
- **Effective Dimensionality** — SVD: minimum principal components explaining 95% of variance
- **Density Distribution** — k-NN distances: coefficient of variation + Local Outlier Factor
- **Retrieval Sensitivity Index** — Jaccard stability of top-k results under calibrated Gaussian perturbation
- **Intrinsic Dimensionality** — TwoNN estimator (Facco et al. 2017): fits d_int from nearest-neighbor distance ratios

---

## Examples

```bash
# Basic audit on synthetic embeddings — runs instantly
python examples/basic_audit.py

# Full pipeline on AG News corpus with sentence-transformers
python examples/full_pipeline.py --n-docs 500 --save-plots ./plots

# Compare geometric health across embedding models
python examples/compare_models.py --n-docs 300
```

Model comparison output:

```
══════════════════════════════════════════════════════════════════════════════
  Spectralyte -- Embedding Model Geometric Health Comparison
══════════════════════════════════════════════════════════════════════════════
  Metric                        all-MiniLM-L6-v2    all-mpnet-base-v2   paraphrase-MiniLM
  ────────────────────────────────────────────────────────────────────────────
  Anisotropy (lower=better)     0.064               0.061               0.057 *
  Effective Dims                150                 165 *               122
  Dim Utilization (higher=better)  39.1% *          21.5%               31.8%
  Density CV (lower=better)     0.039 *             0.044               0.040
  RSI Stability (higher=better) 0.952               0.976 *             0.948
══════════════════════════════════════════════════════════════════════════════
```

---

## Roadmap

- [x] Anisotropy metric
- [x] Effective dimensionality
- [x] Density distribution
- [x] Retrieval Sensitivity Index
- [x] Intrinsic dimensionality (TwoNN)
- [x] Whitening transform
- [x] All-but-the-Top (ABTT) transform
- [x] PCA dimensionality reduction
- [x] Runtime query router
- [x] Matplotlib visualization backend
- [x] Plotly visualization backend
- [x] Remediation plan generator
- [ ] LangChain native integration
- [ ] LlamaIndex native integration
- [ ] Pinecone / Qdrant / Weaviate connectors
- [ ] CLI entrypoint (`spectralyte audit embeddings.npy`)
- [ ] SpectralytePipeline build-time validation gate

---

## Contributing

Contributions are welcome. Please open an issue before submitting a PR.

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
  author  = {Thoppe, Adhviklal},
  title   = {Spectralyte: Illuminating the Geometry of Your Embedding Space},
  year    = {2026},
  url     = {https://github.com/UniqueRed/spectralyte},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.