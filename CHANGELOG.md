# Changelog

All notable changes to mismap-qc. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-11

Adds the first Wave 2 plot, a documentation site, and the repository files and
metadata that were missing for peer review. No breaking changes.

### Added

- **`missing_upset()`** ([#4](https://github.com/foertsch/mismap-qc/issues/4)), the
  first Wave 2 plot. UpSet plot of which sample combinations share missing
  features: for each intersection, how many features are missing in exactly that
  combination. Answers whether particular replicates lose the same features
  together, which bar charts cannot show and Venn diagrams cannot handle past
  three sets. `by="sample"` for one set per sample, or a MultiIndex level name for
  one set per group, where `group_min_frac=0.5` decides when a feature counts as
  lost in a group.

  Needs `upsetplot`, a new optional extra: `pip install mismap-qc[upset]`.

  The plot caps at the 50 largest intersections by default, because intersection
  count grows quickly with sample count. Truncation is annotated on the figure and
  `return_data=True` returns every intersection with a `plotted` column, so
  nothing is silently dropped. Schema: `[feature, members, n_features, rank,
  plotted]`, one row per feature.

- **Documentation site** built with mkdocs-material and mkdocstrings, published to
  GitHub Pages at <https://foertsch.github.io/mismap-qc/> ([#5](https://github.com/foertsch/mismap-qc/issues/5)).
  The API reference generates from the NumPy-style docstrings, so per-function
  parameter tables no longer have to be maintained by hand in the README. Pages:
  home, quickstart, tutorial, API reference across validation / plots / readers,
  contributing, and the generative AI disclosure. New `docs` extra.

- **`CONTRIBUTING.md`** covering the development install, how to run pytest and
  ruff, where code belongs by module, the naming and API conventions, the minimum
  tests a new function needs, the pull request flow, and the release steps.

- **`CODE_OF_CONDUCT.md`** (Contributor Covenant 2.1) with a named reporting
  contact.

- **`CITATION.cff`** so GitHub renders "Cite this repository", including ORCID
  `0000-0003-0409-6209`, plus Citation, Contributing and License sections in the
  README.

- **`docs/generative-ai-use.md`**, disclosing how generative AI was used in
  building this package, as pyOpenSci's generative AI policy requires.

- **A `docs build` CI job** running `mkdocs build --strict`, so a broken internal
  link or a docstring the documentation tooling cannot parse fails the build.

- **A `pytest (all extras)` CI job.** The version matrix installs no optional
  dependencies on purpose, which proves the package works bare but left the
  plotly, anndata and upsetplot paths skipped in CI. The new job installs every
  extra so those actually run.

- **Four guards in `tests/test_package_metadata.py`**: `CITATION.cff`'s version
  field must match `pyproject.toml`, the four repository files the pyOpenSci
  editor check looks for must exist, and the code of conduct must not still carry
  the Contributor Covenant's `[INSERT CONTACT METHOD]` placeholder.

### Changed

- **Maintainer contact is now a personal address** (`foertsch.arion@gmail.com`) in
  package metadata, `CITATION.cff` and the code of conduct. Institutional
  addresses stop resolving when the role ends, and the package may outlive it.
  Affiliation and ORCID still record the institutional link.
- **Author name is now spelled Förtsch**, matching the ORCID record. Published
  0.1.0 and 0.2.x metadata say "Foertsch".
- `[project.urls] Documentation` now points at the documentation site rather than
  the README anchor.
- The sdist allowlist gains `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` and
  `CITATION.cff`. Without them the shipped test suite failed when run from an
  extracted source distribution, which CI does not exercise.
- The commit trailer convention is written down in `CLAUDE.md` and
  `CONTRIBUTING.md`: mark commits a tool wrote, leave the trailer off commits a
  human wrote.

### Fixed

- **`missing_matrix_html()` docstring.** Its `Returns` line sat inside the
  `Parameters` section, so documentation tooling parsed "Returns" as a parameter
  name, and it documented 1 of its 18 parameters. Now has a proper `Returns`
  section and complete parameter documentation. Found by the strict docs build on
  its first run.

## [0.2.2] - 2026-07-30

Packaging release. No functional changes to the library.

### Fixed

- **Source distribution contents.** With no sdist configuration, hatchling swept
  in the entire working tree: the sdist was 2.2 MB and shipped
  `examples/output/*.png` (879 KB for one file), the CPTAC notebook, `uv.lock`,
  `CLAUDE.md`, `DIARY.md`, the pre-rename `pretty_missing.py` shim, and
  `.claude/settings.local.json`. `[tool.hatch.build.targets.sdist]` now declares
  an explicit allowlist: the package, the test suite, README, CHANGELOG, LICENSE,
  and `pyproject.toml`. The wheel was never affected.

  `tests/` is included deliberately, so downstream packagers can verify a build.

## [0.2.1] - 2026-07-30

Packaging and metadata release. No functional changes to the library; this is
the first PyPI release that carries the 0.2.0 validation API.

### Fixed

- **Repository URL on PyPI.** The published 0.1.0 metadata pointed at
  `github.com/afoertsch/mismap-qc`, which does not exist. Corrected in the repo
  during 0.2.0 development but never published, and this release ships the fix.
- **Package description.** Was "Missing-data *matrix* for RNA-Seq and proteomics
  QC", which framed the package as visualization. Now reads "Missing-data
  *validation* for proteomics and RNA-Seq QC", matching the README and the
  package's actual scope.
- **CHANGELOG 0.2.0 notes** claimed the single-file design was preserved. 0.2.0
  *is* the package split; the note contradicted the release it documented.

### Added

- Author email in package metadata.
- `Documentation`, `Issues`, and `Changelog` entries under `[project.urls]`.
- Python 3.13 to the CI test matrix and the trove classifiers.
- `Operating System :: OS Independent` and
  `Topic :: Scientific/Engineering :: Information Analysis` classifiers.
- `ruff` to the `dev` extra, so `pip install -e ".[dev]"` provides both CI tools.
- `tests/test_package_metadata.py`, guarding version agreement between
  `pyproject.toml` and `mismap_qc.__version__`, and the presence of the metadata
  the packaging guidelines require.

### Changed

- Keywords: `visualization` replaced with `data-validation`.
- Dropped the `Topic :: Scientific/Engineering :: Visualization` classifier.

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

- **Package split.** The single-file `mismap_qc.py` (3,031 lines) was refactored
  into a `mismap_qc/` package with seven submodules (`_core`, `stats`,
  `validation`, `plots`, `io`, `lod`, `__init__`). The public API is unchanged:
  `from mismap_qc import qc, missing_matrix, ...` works as before.
- Developed on `feat/validation-api` over nine commits and squash-merged as
  `5cd2f0a` ([#2](https://github.com/foertsch/mismap-qc/pull/2)); the PR retains
  the per-checkpoint history.
- Never published to PyPI. 0.2.1 is the first PyPI release to carry the
  validation API.
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
