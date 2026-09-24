# Changelog

## v0.3.0 (2026-09-23)

### Added
- CLI entrypoint (`spectralyte`) with `audit`, `fix-plan`, `transform`, and
  `version` subcommands. `audit --json` emits the same schema as
  `report.export()` on clean stdout, so it composes with `jq` and CI gates.

### Changed
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
- 478 tests passing across all modules, including regression coverage for the
  grading fix above.
- Dedicated property tests for the three correction transforms
  (`tests/test_transforms/`), covering the mathematics rather than just
  shapes: whitening flattens the covariance spectrum, ABTT genuinely projects
  out the top-k principal directions, and `pca_reduce` returns decorrelated
  components.
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