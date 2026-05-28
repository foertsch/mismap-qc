# Changelog

All notable changes to mismap-qc. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-28

The first release with a programmatic validation API. The package now exposes
both a "validate this dataset" entry point and the per-check visualizations
that pair with it.

### Added

- **Validation API.** `qc(df)` runs a battery of missing-data checks and
  returns a frozen `MismapReport`. `assert_qc(df, thresholds=...)` raises
  `MismapQCFailure` on rule violation. `report.check(thresholds=...)` and
  `report.passes(...)` are no-raise alternatives.
- **11 threshold rules.** `min_sample_completeness`,
  `min_sample_completeness_per_group`, `min_features_detected`,
  `max_feature_missing_rate`, `max_sample_outliers`,
  `max_sample_outlier_zscore`, `max_mnar_fraction`,
  `max_unclassified_fraction`, `min_group_completeness`,
  `max_batch_effect_features`, `max_runorder_slope`. Each rule has a default
  severity (error / warning / info) that callers can override via
  `severity_overrides`.
- **`MismapQCWarning`.** Warning-severity rule violations emit
  `MismapQCWarning` via the standard `warnings` module. Silenceable with
  `warnings.filterwarnings("ignore", category=MismapQCWarning)`.
- **`MismapReport` serialization.** `to_dict()`, `to_json()`, `to_html()`,
  plus `__repr__` and `summary()` for one-line and multi-section views.
- **`missing_mechanism()`** classifies per-feature missingness as
  MNAR / MAR / MCAR / INSUFFICIENT via one-sided Mann-Whitney U on
  per-sample mean abundance. Returns `(Figure, DataFrame)`.
- **`comissing_heatmap()`** plots pairwise co-missingness for the top-N
  most-missing features with optional hierarchical clustering.
- **`return_data=True`** flag on every plot function. When set, returns
  `(Figure, DataFrame)` with a documented schema. Schemas are registered in
  `_RETURN_DATA_SCHEMAS` and protected by a regression test.
- **`from_anndata()`** reads an AnnData object into the features × samples
  DataFrame mismap-qc expects. Supports `obs_levels`, `var_index`, `layer`,
  and three `missing_value` strategies (`"nan"`, `"zero"`, or float
  threshold). `anndata` is an optional dependency
  (`pip install mismap-qc[anndata]`).
- **`estimate_lod()`** per-feature limit-of-detection estimation with
  `method="min"` or `method="quantile"`.

### Changed

- README rewritten around the validation framing. First runnable example is
  now `qc()` / `assert_qc()` rather than `missing_matrix()`. Adds a "What
  this validates" table and a "Why use this" Statement of Need.
- The pre-existing `missing_mechanism()` / `sample_outlier_score()` /
  `batch_missing_test()` API sketches from `docs/PLAN_new_plots.md`
  consolidated so that the analytical helpers (`_classify_mechanism`,
  `_top_codropouts`, `_batch_missing_test`, etc.) are shared between `qc()`
  and the plot functions rather than duplicated.

### Notes

- Single-file design preserved. `mismap_qc.py` is ~2900 lines.
- `feat/validation-api` branch contains the five checkpoint commits that
  produced this release. See git log for the per-checkpoint scope.
- Wave 2 plots (`missing_upset`, `sample_outlier_score`, `batch_missing_test`,
  `missing_summary_report`), additional Scope E items
  (`imputation_diagnostic`, `replicate_concordance`), search-engine output
  parsers (MaxQuant / DIA-NN / FragPipe / Spectronaut), and the CLI are
  deferred to a later release.

## [0.1.0] - 2026-03-11

Initial release.

### Added

- `missing_matrix()` static nullity matrix with hierarchical clustering and
  MultiIndex annotation strips.
- `missing_matrix_html()` Plotly-based interactive version.
- `missing_abundance_density()` companion plot.
- `completeness_bars()` per-group completeness bars.
- `detection_waterfall()` feature-detection threshold curve.
- `missing_runorder()` missingness over run order / time.
- CPTAC LUAD proteomics tutorial notebook.
- GitHub Actions CI on Python 3.10-3.12, ruff lint job, macOS test matrix.
