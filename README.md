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
  Intrinsic Dimension    42.3   (R²=0.991)   ✓  MODERATE
════════════════════════════════════════════════════
  ✓ No issues detected. Embedding space looks healthy.
```

---

## Why Spectralyte

Measuring retrieval quality directly requires labelled data — queries with
known-correct documents. Most teams don't have that, and building it is weeks
of annotation nobody budgets for.

Spectralyte needs no labels. It reads your embedding matrix and tells you
whether the space itself is structurally broken.

It will usually tell you it isn't. Modern encoders are trained with contrastive
objectives that produce healthy geometry by construction, and on a healthy
space these corrections make retrieval *worse* — so Spectralyte declines to
recommend them. That verdict is the product: a cheap, label-free check that your
embedding pipeline is not silently misconfigured.

When something *is* wrong — raw LM hidden states used as embeddings, the wrong
pooling, a fine-tune that collapsed — it finds it and fixes it. On mean-pooled
GPT-2 embeddings, whitening lifted nDCG@10 from 0.028 to 0.319. See
[Does it actually work?](#does-it-actually-work) for how that was measured.

**What it is not:** a measure of retrieval quality. A healthy verdict means the
geometry is sound, not that your search is good. Chunking, domain mismatch,
reranking and data quality all sit outside what geometry can see.

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
fixed = audit.transform(strategy="whiten")

# Fix anisotropy via All-but-the-Top (ABTT)
fixed = audit.transform(strategy="abtt", abtt_k=3)

# Reduce to effective dimensionality (saves storage, speeds retrieval)
fixed = audit.transform(strategy="pca_reduce")

# Re-audit to verify improvement
report_fixed = audit.run(fixed)
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

### Transforming queries

The transform is fitted once on the audited corpus and applied unchanged
afterwards, so a query maps exactly where that vector would map in the index.
**Send every query through the same call before searching** — an untransformed
query and a transformed index are in different spaces, and retrieval degrades
without raising anything.

```python
audit = Spectralyte(corpus)
audit.run()

index_vectors = audit.transform(strategy="whiten")   # defaults to the corpus
# ... index index_vectors ...

# At query time — a single (d,) vector is fine, and comes back 1D
query_vectors = audit.transform(query_embedding, strategy="whiten")
```

The fit lives on the `Spectralyte` instance. To use it from another process —
a serving loop, a different machine — save it:

```python
audit.fitted_transform().save("spectralyte_fit.npz")

# ... later, wherever queries are embedded ...
from spectralyte.core.transform import FittedTransform
fit = FittedTransform.load("spectralyte_fit.npz")
query_vectors = fit.apply(query_embedding, strategy="whiten")
```

The file is plain arrays (`numpy.savez`, no pickle), so loading one cannot
execute code. Re-running `run()` on a different matrix establishes a new
reference space and refits.

> **Whitening is not always the right fix.** It rescales every direction to
> equal variance, which on an ill-conditioned space amplifies near-null
> directions and can make retrieval worse than leaving the embeddings alone.
> Check `report.compare()` against a retrieval metric you trust before shipping
> it; `abtt` is the more conservative choice.

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

Issue 1: High Density Clustering (CV=0.84)  [CRITICAL]
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

## CLI

Spectralyte ships a command-line interface for auditing embeddings without writing any Python.

### Audit

```bash
# Human-readable summary
spectralyte audit my_embeddings.npy

# Structured JSON — pipe to jq, CI gates, or any downstream tooling
spectralyte audit my_embeddings.npy --json

# Override configuration
spectralyte audit my_embeddings.npy --config k=10 variance_threshold=0.90
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
  Intrinsic Dimension    42.3   (R²=0.991)   ✓  MODERATE
════════════════════════════════════════════════════
  ✓ No issues detected. Embedding space looks healthy.
```

### Remediation plan

```bash
spectralyte fix-plan my_embeddings.npy
spectralyte fix-plan my_embeddings.npy --framework langchain
spectralyte fix-plan my_embeddings.npy --framework llamaindex
```

### Transform

```bash
# Fix anisotropy — saves corrected embeddings to fixed.npy
spectralyte transform my_embeddings.npy --strategy whiten --output fixed.npy

# All-but-the-Top
spectralyte transform my_embeddings.npy --strategy abtt --output fixed.npy

# Reduce to effective dimensionality
spectralyte transform my_embeddings.npy --strategy pca_reduce --output reduced.npy

# Keep the fitted transform so queries can reach the same space
spectralyte transform corpus.npy --strategy whiten \
    --output fixed.npy --save-fit fit.npz

# At query time — apply the saved fit, never refit
spectralyte transform queries.npy --strategy whiten \
    --apply-fit fit.npz --output queries_fixed.npy
```

### JSON output

The `--json` flag emits the same schema as `report.export()`, making it composable with downstream tooling:

```bash
# Pipe into jq
spectralyte audit embeddings.npy --json | jq '.anisotropy.score'

# Save to file
spectralyte audit embeddings.npy --json > audit.json

# CI gate: fail if issues detected
spectralyte audit embeddings.npy --json | python -c "
import sys, json
data = json.load(sys.stdin)
if data['n_issues'] > 0:
    print(f\"{data['n_issues']} geometry issues detected\")
    sys.exit(1)
"
```

### Version

```bash
spectralyte version
```

---

## Metric tiers

The default audit computes two metrics. Both are validated: on two BEIR
datasets across three encoders they alone predicted whether a transform would
help retrieval, correctly on 6/6 pairs.

| tier | metric | in the verdict? |
|---|---|---|
| **core** | Anisotropy | yes |
| **core** | Effective dimensionality | yes |
| experimental | Density distribution | no |
| experimental | Retrieval Sensitivity (RSI) | no |
| experimental | Intrinsic dimensionality (TwoNN) | no |

The experimental three measure real geometric properties, but none has a
demonstrated relationship to retrieval quality — `intrinsic_dim` rated every
space in the benchmark "low", including encoders scoring nDCG@10 0.656. They
never affect `n_issues` or `needs_transform`, and they are off by default
because two of them dominate the runtime.

```python
report = audit.run()                      # core only — 3.2x faster
report = audit.run(experimental=True)     # all five
```

```bash
spectralyte audit embeddings.npy --experimental
```

`get_router()` requires `experimental=True`, since the router is built from the
density and sensitivity metrics.

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

## Does it actually work?

`benchmarks/retrieval_benchmark.py` tests the premise directly — that bad
geometry hurts retrieval and these transforms fix it — on two BEIR datasets
crossed with three encoders, scored by nDCG@10 against human relevance
judgments.

`needs_transform` agreed with the measured outcome on **6/6 dataset-model
pairs**:

| dataset | encoder | nDCG@10 | `needs_transform` | after whitening |
|---|---|---|---|---|
| SciFact | all-mpnet-base-v2 | 0.6557 | no | 0.6326 (−0.023) |
| SciFact | all-MiniLM-L6-v2 | 0.6451 | no | 0.5862 (−0.059) |
| SciFact | **gpt2 mean-pooled** | 0.0284 | **yes** | **0.3185 (+0.290)** |
| NFCorpus | all-mpnet-base-v2 | 0.3347 | no | 0.2302 (−0.105) |
| NFCorpus | all-MiniLM-L6-v2 | 0.3177 | no | 0.2491 (−0.069) |
| NFCorpus | **gpt2 mean-pooled** | 0.0148 | **yes** | **0.0634 (+0.049)** |

On a genuinely pathological space, whitening lifted nDCG@10 by **11.2x**
(SciFact) and **4.3x** (NFCorpus). On every healthy space every transform made
retrieval *worse* — so Spectralyte declining to recommend one matters as much
as it recommending one.

The MiniLM/SciFact baseline matches the published BEIR figure (~0.645), so the
harness is comparable to the literature.

**Read the caveats** in [`benchmarks/README.md`](benchmarks/README.md): two
datasets and one clearly-pathological encoder is a small sample, `density`,
`sensitivity` and `intrinsic_dim` are not validated by it, and you should still
check against your own retrieval metric before mutating a production index.

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
- [x] CLI entrypoint (`spectralyte audit embeddings.npy`)
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