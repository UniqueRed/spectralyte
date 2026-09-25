# Validation

Spectralyte claims that bad embedding geometry hurts retrieval and that its
transforms fix it. That link is the whole premise, and it is not self-evident —
the metrics measure geometry while the promise is about search quality. This
directory contains the benchmark that tests it, so the claim can be checked
rather than taken on faith.

```bash
pip install "spectralyte[dev]" sentence-transformers datasets torch
python benchmarks/retrieval_benchmark.py --out results.json
```

Two BEIR datasets crossed with three encoders, scored by nDCG@10 against human
relevance judgments. Exact cosine search, no approximate index, so the numbers
isolate the geometry.

- **SciFact** — 5,183 scientific abstracts, 300 judged claims
- **NFCorpus** — 3,633 medical documents, 323 judged queries
- **Encoders** — mean-pooled GPT-2 (the textbook anisotropic space, per
  Ethayarajh 2019), `all-MiniLM-L6-v2`, `all-mpnet-base-v2`

## The headline result

`needs_transform` agreed with the measured outcome on **6/6 dataset-model
pairs**: it recommended a transform exactly where one helped, and declined
everywhere it would have hurt.

| dataset | encoder | anisotropy | dims | nDCG@10 | `needs_transform` | after whitening |
|---|---|---|---|---|---|---|
| SciFact | all-mpnet-base-v2 | healthy | healthy | 0.6557 | no | 0.6326 (−0.023) |
| SciFact | all-MiniLM-L6-v2 | healthy | healthy | 0.6451 | no | 0.5862 (−0.059) |
| SciFact | **gpt2 mean-pooled** | **severe** | **critical** | 0.0284 | **yes** | **0.3185 (+0.290)** |
| NFCorpus | all-mpnet-base-v2 | moderate | healthy | 0.3347 | no | 0.2302 (−0.105) |
| NFCorpus | all-MiniLM-L6-v2 | moderate | healthy | 0.3177 | no | 0.2491 (−0.069) |
| NFCorpus | **gpt2 mean-pooled** | **severe** | **critical** | 0.0148 | **yes** | **0.0634 (+0.049)** |

On the pathological space whitening lifted nDCG@10 by **11.2x** on SciFact and
**4.3x** on NFCorpus. On every healthy space, every transform made retrieval
worse — which is why `needs_transform` declining matters as much as it firing.

The MiniLM/SciFact baseline of 0.6451 matches the published BEIR figure
(~0.645), so these numbers are comparable to the literature and the harness is
not flattering itself.

## What this does not show

- **Two datasets and three encoders.** The separation is clean but the sample
  is small. A production decision on your own corpus should be checked against
  your own retrieval metric.
- **Only one clearly pathological encoder.** GPT-2 mean-pooling is a
  deliberately extreme case. Where the boundary sits between "moderate" and
  "act on it" is not settled by this data — it is only shown that `moderate`
  alone is not grounds to transform.
- **nDCG@10 with exact search.** An approximate index adds its own error, and
  a reranker downstream may absorb geometry problems this benchmark attributes
  to the encoder.
- **No claim about `density`, `sensitivity`, or `intrinsic_dim`.** Those
  metrics are not validated here. `intrinsic_dim` in particular showed *no*
  retrieval-predictive power (see below).

## Calibrations this benchmark corrected

Three of the library's defaults were set from synthetic Gaussian fixtures and
were wrong on real embeddings. All three were changed on this evidence.

**`whiten_rcond`: 0.01 → 0.0001.** The whitening eigenvalue floor was far too
aggressive, discarding most of the available gain. Both pathological cases peak
at exactly `1e-4`:

| `whiten_rcond` | 1e-12 | 1e-6 | 1e-5 | **1e-4** | 1e-3 | 1e-2 | 5e-2 |
|---|---|---|---|---|---|---|---|
| SciFact / gpt2 | +0.222 | +0.229 | +0.255 | **+0.290** | +0.180 | +0.047 | +0.008 |
| NFCorpus / gpt2 | +0.025 | +0.027 | +0.046 | **+0.049** | +0.020 | +0.001 | −0.002 |

The old default of `1e-2` captured +0.047 of an available +0.290.

**`needs_transform` now requires a BAD grade, not merely WARN.** Firing on
`moderate` anisotropy produced two false positives — NFCorpus/MiniLM and
NFCorpus/mpnet — where it advised a transform that cost 0.07-0.10 nDCG.
Requiring BAD took the advice from 4/6 to 6/6 correct.

**A condition-number heuristic was removed.** It steered ill-conditioned spaces
toward ABTT. But healthy encoders here have condition numbers of 6.9e34 to
3.8e38 and GPT-2 has 1.5e14, so a 1e4 threshold fires on everything and
discriminates nothing — and whitening beat ABTT on both cases that mattered
(0.319 vs 0.274, 0.063 vs 0.059).

**`intrinsic_dim` "low" was demoted from BAD to WARN.** Every space measured
here reports "low", including the two best encoders, and GPT-2 scores a *higher*
intrinsic dimension (18.6-19.4) than mpnet (8.1-10.5). Low intrinsic
dimensionality is normal for sentence embeddings, not a pathology, and treating
it as critical put a false "issue detected" on an encoder scoring 0.6451.
Genuine manifold collapse still trips `dimensionality`, which does grade BAD.

## Choosing `abtt_k`

The default of 3 is tuned for safety, not for severely anisotropic spaces. On
GPT-2/SciFact, ABTT improves monotonically with `k` across the range tested
(+0.005 at k=1, +0.129 at k=10, +0.245 at k=40), while on healthy encoders every
`k` hurts and larger `k` hurts more. If you are treating a badly anisotropic
space with ABTT rather than whitening, sweep `k` rather than trusting the
default.
