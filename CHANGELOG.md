# Changelog

## v0.3.0 (2026-08-31)

### Added
- CLI entrypoint (`spectralyte`) with `audit`, `fix-plan`, `transform`, and
  `version` subcommands. `audit --json` emits the same schema as
  `report.export()` on clean stdout, so it composes with `jq` and CI gates.

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

### Removed
- Empty `spectralyte.integrations` package (never imported; the LangChain and
  LlamaIndex integrations remain on the roadmap).

### Internal
- 430 tests passing across all modules, including regression coverage for the
  grading fix above.

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