# Changelog

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