# Changelog

## v0.4.0 (2026-09-25)

Spectralyte now ships what the benchmark validated and makes the rest opt-in.

### Changed
- **The default audit computes two metrics, not five.** Anisotropy and
  effective dimensionality are the pair that predicted, on 6/6 dataset-model
  pairs, whether a correction transform would help retrieval. Density,
  retrieval sensitivity and intrinsic dimensionality measure real geometric
  properties but none has a demonstrated link to retrieval quality, so they are
  now computed only via `run(experimental=True)` / `--experimental`.

  They also dominated the runtime: on 5000x384 the default audit is **3.2x
  faster** (7.0s vs 22.4s).
- **Only core metrics decide the verdict.** `n_issues` and `needs_transform`
  ignore the experimental metrics entirely, so an unvalidated signal can no
  longer raise a false alarm on a healthy index. Experimental findings surface
  separately via `experimental_findings`, in their own block in `summary()`,
  and tagged as leads rather than diagnoses in `fix_plan()`.
- `has_brittle_zones` returns `None` when sensitivity was not computed. `None`
  means unmeasured, not "no" — returning `False` would assert something never
  checked.
- `export()` and `audit --json` omit experimental sections when absent rather
  than emitting nulls a CI gate might misread. The two payloads are still
  asserted identical by the test suite.
- `plot()` renders only the metrics the report carries.

### Added
- `AuditReport.experimental_findings`, `.has_experimental`, `.measured_metrics`.
- `severity.CORE_METRICS` and `severity.EXPERIMENTAL_METRICS`.

### Notes
- `get_router()` now requires `run(experimental=True)`, since the router is
  built from density and sensitivity. It raises with that instruction rather
  than failing obscurely.
- Nothing was deleted. Every metric, the router and all their tests remain;
  they are opt-in rather than default. The full five-metric release is tagged
  `v0.3.0` and branched as `v0.3-full`.

## v0.3.0 (2026-09-23)

### Added
- CLI entrypoint (`spectralyte`) with `audit`, `fix-plan`, `transform`, and
  `version` subcommands. `audit --json` emits the same schema as
  `report.export()` on clean stdout, so it composes with `jq` and CI gates.

### Added
- `benchmarks/retrieval_benchmark.py` — the validation this library was missing.
  Two BEIR datasets (SciFact, NFCorpus) crossed with three encoders, scored by
  nDCG@10 against human relevance judgments. `needs_transform` agreed with the
  measured outcome on 6/6 dataset-model pairs: whitening lifted nDCG@10 11.2x on
  mean-pooled GPT-2 / SciFact (0.028 to 0.319) and 4.3x on NFCorpus, while every
  transform hurt every healthy encoder. The MiniLM/SciFact baseline reproduces
  the published BEIR figure. Caveats in `benchmarks/README.md`.
- `spectralyte.core.transform.FittedTransform`, the fitted parameters as an
  object that outlives the auditor, with `save()` / `load()`. Stored as plain
  arrays via `numpy.savez` — no pickle, so loading a transform produced
  elsewhere cannot execute code.
- `spectralyte transform --save-fit FIT.NPZ` and `--apply-fit FIT.NPZ`. Without
  these the CLI produced an index nobody could query correctly: the fitted
  mapping died with the process, leaving no way to put a query into the
  corpus's space. Transforming without `--save-fit` now says so on stderr.
- `AuditReport.condition_number`, the ratio of largest to smallest variance
  across dimensions, derived from the dimensionality spectrum at no extra cost.

### Changed
- **`whiten_rcond` default is 1e-4, set by benchmark rather than by intuition.**
  An earlier 0.01, chosen from synthetic fixtures, damped far too hard: it
  captured +0.047 nDCG@10 of an available +0.290 on SciFact and +0.001 of
  +0.049 on NFCorpus. Both pathological cases peak at exactly 1e-4.
- **`needs_transform` now requires a BAD grade rather than merely WARN.** Firing
  on `moderate` anisotropy advised transforms that cost 0.07-0.10 nDCG@10 on two
  healthy encoders. Requiring BAD took the advice from 4/6 to 6/6 correct.
- **`intrinsic_dim` "low" is graded WARN, not BAD.** Every real space measured
  reports "low", including encoders at nDCG@10 0.656, and mean-pooled GPT-2
  scores a *higher* intrinsic dimension than either sentence encoder. The label
  showed no retrieval-predictive power, so it no longer drives `n_issues` or
  `needs_transform` on its own. Genuine collapse still trips `dimensionality`.
- **Removed the condition-number heuristic that steered ill-conditioned spaces
  to ABTT.** Real embedding spaces all have enormous condition numbers (6.9e34
  to 3.8e38 for healthy encoders), so the threshold fired on every space and
  discriminated nothing — and whitening beat ABTT on both cases that mattered.
- **Whitening floors covariance eigenvalues relative to the largest**
  rather than at an absolute `1e-10`. Whitening
  scales each direction by `lambda^-1/2`, so an absolute floor let a near-null
  direction be amplified roughly 1e5x — drowning the signal in noise. Measured
  on a corpus with condition number 2.6e6, recall@1 went from 1.00 unmodified
  to 0.00 whitened; with the relative floor it holds at 1.00. The floor is
  inert on a well-conditioned spectrum, where nothing sits below it.
- **`fix_plan()` no longer recommends whitening for an ill-conditioned space.**
  Anisotropy and a fast-decaying spectrum co-occur in real embedding models, so
  the previous advice was most dangerous exactly where it was most likely to be
  followed. Above a condition number of 1e4 the plan recommends ABTT, explains
  why, and points at `whiten_rcond` for anyone who still wants whitening.
- **`transform()` now fits once and applies many times.** Every strategy
  previously recomputed its transform from whatever array it was handed, so
  there was no way to put a query into the same space as the index — the
  central remediation workflow. A single query centered against itself became
  the zero vector, silently: all three strategies returned all-zeros for a
  one-row input, and following the documented "transform incoming query
  embeddings the same way" drove recall@1 to 0.000 in a 2000-document
  benchmark. Batched calls fared better only by accident, and their output
  shifted with the composition of the batch.

  The transform parameters (corpus mean, whitening matrix, right singular
  vectors) are now fitted lazily against the audited matrix and cached, then
  applied unchanged. `transform()` accepts a single `(d,)` vector and answers
  in kind, defaults to the audited corpus when given nothing, rejects a width
  that does not match the audit, and refits when `run()` is called on a new
  matrix. Per-query and batched output are now bit-identical.

### Fixed
- **Inverted health grading for intrinsic dimensionality.** `report.n_issues`,
  `summary()`, the exported JSON, and both plot backends all graded
  interpretation labels against one shared set that treated `"low"` as healthy.
  For intrinsic dimensionality `"low"` means the manifold has collapsed
  (`d_int / nominal_dims < 0.05`) — the pathological case — so collapsed spaces
  were reported clean while healthy full-rank spaces were flagged. The same set
  also masked `dimensionality == "low"` (4-10% utilization), which ranked as
  healthier than the `"moderate"` tier above it.
- Metric results are now graded against their own metric's polarity via the new
  `spectralyte.core.severity` module; the interpretation labels are not
  comparable across metrics and are no longer treated as if they were.
- **`fix_plan()` and `n_issues` disagreed.** `n_issues` counts any metric that
  is not healthy, but `fix_plan()` carried a hand-maintained if-chain that
  skipped the `moderate` tier for dimensionality, density and sensitivity, and
  had no branch at all for intrinsic dimensionality. A report could print
  "3 issues detected — run fix_plan()" and then return a plan addressing one of
  them; a collapsed manifold, the most serious finding available, produced no
  guidance whatsoever. The plan now iterates the same severity grading
  `n_issues` uses, so the two cannot drift apart, and tags each section
  CRITICAL or WARNING rather than presenting a borderline reading with the
  same urgency as an active failure.
- Added a remediation section for collapsed manifolds, which states plainly
  that no transform fixes them — whitening and ABTT redistribute variance and
  cannot recreate information that was never present — and points at corpus
  duplication, truncating preprocessing, and model mismatch instead.
- `needs_transform` is likewise graded through `severity` instead of its own
  label thresholds.
- Remediation plan issue numbering. `fix_plan()` hardcoded a number per issue
  *type*, so a report with no anisotropy problem opened at "Issue 2" and read
  like a truncated document. Issues are now numbered in emission order.
- Redundant SVD in the ABTT transform. `_abtt` computed a full SVD for the left
  singular vectors, discarded them unused, then computed a second full SVD on
  the same matrix for the right singular vectors it actually needs. Removing the
  dead first pass halves the cost of every `strategy="abtt"` call; output is
  bit-identical.

### Removed
- Empty `spectralyte.integrations` package (never imported; the LangChain and
  LlamaIndex integrations remain on the roadmap).
- Empty `spectralyte.utils` package (`preprocessing` and `sampling` were both
  zero-byte placeholders that nothing imported).

### Internal
- 500 tests passing across all modules, including regression coverage for the
  grading fix above.
- Dedicated property tests for the three correction transforms
  (`tests/test_transforms/`), covering the mathematics rather than just
  shapes: whitening flattens the covariance spectrum, ABTT genuinely projects
  out the top-k principal directions, and `pca_reduce` returns decorrelated
  components.
- Tests for the whitening floor (inert when well-conditioned, protective when
  not, and bounding the amplification), for conditioning-aware remediation, and
  for fit persistence including a check that the format stays pickle-free.
- Contract tests for the fit/apply split: a lone query must match its row in
  the transformed corpus, output must not depend on batch composition, and a
  stale fit must not survive a new audit. Mutation testing confirms they fail
  if the per-call refit is reintroduced.
- Regression tests locking the `fix_plan()` / `n_issues` invariant: the plan
  must emit exactly one section per counted issue, numbered sequentially, with
  a collapsed manifold always producing guidance.
- Lint is clean (`ruff check .` passes): removed unused imports and variables,
  dropped placeholder-less f-strings, and reordered the Plotly test imports so
  the `importorskip` guard no longer sits mid-import-block.

## v0.2.0 (2026-04-16)

### Added
- Five geometric metrics: Anisotropy, Effective Dimensionality, Density
  Distribution, Retrieval Sensitivity Index, Intrinsic Dimensionality (TwoNN)
- AuditReport with summary(), plot(), compare(), fix_plan(), export()
- Spectralyte orchestrator wiring all five metrics
- Three direct transforms: whiten, ABTT, pca_reduce
- Runtime router with centroid-based zone classification
- Dual visualization backends: matplotlib (default) and plotly
- Example scripts: basic_audit.py, full_pipeline.py, compare_models.py
- 338 tests passing across all modules

## v0.1.0 (2026-04-01)

### Added
- Initial PyPI placeholder release
- Package structure and CI/CD pipeline