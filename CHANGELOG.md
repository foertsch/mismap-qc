# Changelog

All notable changes to mismap-qc. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`missing_upset()` crashed on pandas 3** with `ValueError: Invalid RGBA
  argument: nan`, in 0.3.x and 0.4.0 alike. upsetplot 0.9.0, the latest release,
  fills unset dot styles with in-place `fillna` calls on DataFrame columns, which
  pandas 3's copy-on-write no longer writes back
  ([jnothman/UpSetPlot#303](https://github.com/jnothman/UpSetPlot/issues/303)).
  `missing_upset()` now applies those defaults itself for the duration of the
  matrix draw. The figure is pixel-identical on pandas 2 and 3. The four
  `ChainedAssignmentError` warnings upsetplot emits per call on pandas 3 are
  suppressed along with its pandas 2 FutureWarnings.

## [0.4.0] - 2026-09-28

Two checks now give different results on the same data: sample outlier
detection and the missingness mechanism classifier. Both were wrong on small
designs, and both are described under Changed, along with a new default for
the smoother in `missing_runorder()`. Read that section before upgrading if you
have thresholds tuned against 0.3.x. New in this release: `completeness_violin()`.

```bash
pip install --upgrade mismap-qc
```

### Added

- **Static type checking with mypy**, run in CI (`typecheck` job) and configured
  under `[tool.mypy]`. mypy is pinned to 2.3.1 in the workflow, matching the ruff
  approach, so a checker release cannot change what passes without a deliberate
  bump.

- **Test coverage** measured in CI with `pytest-cov` and reported to Codecov,
  with a badge in the README. Measured in the all-extras job, the only one that
  reaches the optional-dependency code paths.

- **An `sdist` CI job** that builds the source distribution, extracts it, and runs
  the bundled test suite from the extracted copy. Every other job tests the
  working tree, so a file missing from the sdist allowlist could pass CI and still
  ship a broken test suite, which happened once in 0.2.x and was caught by hand.
  Verified by dropping `CITATION.cff` from the allowlist: the job fails.

- **Python 3.14** in the CI test matrix and the trove classifiers. The suite passes
  on 3.14 both bare and with every optional extra installed.

- **`completeness_violin()`**, the distribution behind `completeness_bars()`.
  A bar shows one mean per group; a violin shows every sample's completeness in
  the group, so a single bad run is visible instead of averaged away.
  `level="features"` shows each feature's detection rate within the group
  instead. Groups are ordered and coloured as in `completeness_bars()`, and
  `return_data=True` returns one row per sample or feature (`group`, `member`,
  `value`, `level`).

### Changed

- **Sample outlier detection now uses a robust z-score. This changes results.**
  The previous check used a classic z-score (`|z| > 2.5`), which could not flag
  any sample in a group of eight or fewer. An outlier inflates the standard
  deviation it is measured against, capping |z| at (n - 1) / sqrt(n): 1.79 for
  five replicates, so a replicate missing 90% of its features went unflagged.
  Most proteomics designs have three to eight replicates per condition.

  A sample is now flagged when, within its group, its robust z-score (median and
  median absolute deviation, after Iglewicz & Hoaglin) exceeds 3.5 **and** its
  missing rate is at least 0.10 above the group median. Requiring both avoids
  false alarms in either direction: the z-score alone flags trivially worse
  samples in groups that agree closely, and the gap alone flags ordinary
  variation in noisy ones.

  Consequences for existing users:

  - **`max_sample_outlier_zscore` now compares against the robust z-score**, which
    runs on a different scale. A threshold set for the old score means something
    else now.
  - **Flagging is one-sided.** Only samples worse than their peers are flagged,
    and `max_sample_outlier_zscore` measures the same direction. A sample with
    unusually *low* missingness no longer fails it.
  - **Groups under three samples are reported as not evaluable** rather than as
    having no outliers. Their `z_score` is NaN, the new `evaluable` column is
    False, both outlier rules are skipped, and the report says "outliers not
    evaluable". Previously these groups silently reported zero outliers.
  - The `max_sample_outlier_zscore` failure message lists only samples above the
    threshold. It previously listed the three highest scores whether or not they
    crossed it.

- **New `qc()` options** `outlier_z_threshold` (default 3.5) and
  `outlier_min_delta` (default 0.10) to tune the two conditions.

- **Missingness mechanism: features the test cannot decide are now
  INSUFFICIENT, not MAR. This changes results.** With every detected sample
  outranking every missing one, the one-sided Mann-Whitney p-value is
  1 / C(n, k) for n samples of which k are missing. For three detected against
  three missing that is 1/20 = 0.05, which fails `p < 0.05`, so even a perfectly
  separated protein was labelled MAR, a conclusion the data cannot support.
  Such features, and any others whose best achievable p-value cannot reach
  `alpha`, are now INSUFFICIENT with a NaN p-value.

  This also closes a false positive. With a tied sample mean, scipy drops the
  exact test for a normal approximation, which reported p = 0.038 for three
  against three and called the feature MNAR. The same check now rules it
  INSUFFICIENT.

  At the default `alpha=0.05` only the three-against-three case is affected. A
  stricter `alpha` in `missing_mechanism()` marks more small designs
  INSUFFICIENT, because they cannot reach it either. Expect
  `max_unclassified_fraction` to rise and the denominator of
  `max_mnar_fraction` to fall on small datasets.

- The `missing_mechanism()` docstring listed "MCAR" as a possible result. The
  classifier has never produced it: "MAR" means no evidence of
  abundance-dependent dropout and covers both. Corrected.

- **`missing_runorder()` smooths within each group.** With `group_level` set,
  the rolling mean used to run over all samples in run order, so for two
  acquisition batches it averaged across the boundary and drew one continuous
  drift line through both sessions. Each group now gets its own line. The line
  also breaks wherever the gap between consecutive run-order values is more than
  ten times the median gap, instead of drawing a flat segment through run order
  with no samples in it. This changes the plot only: `qc()` measures run-order
  drift separately and its results are unaffected.

### Deprecated

- **`min_sample_completeness_per_group`**, to be removed in a future release.
  It always returns the same value as `min_sample_completeness`: the lowest
  completeness among each group's lowest is the lowest overall. Use
  `min_sample_completeness` instead. The rule keeps working and returning its
  existing value until then, so no thresholds change meaning, and using it
  emits a `DeprecationWarning`.

### Fixed

- **`missing_mechanism()` no longer draws an empty MCAR bar.** Its bar chart had
  a category for MCAR that was always zero, because the classifier never
  separates MCAR from MAR. A bar that is always zero reads as "tested for MCAR
  and found none". The chart now shows only the labels the classifier can
  return: MNAR, MAR and INSUFFICIENT.
- **Return-type annotations on the six `return_data` plot functions.** They were
  annotated `-> plt.Figure` but return `(Figure, DataFrame)` when
  `return_data=True`. The signatures now read
  `-> plt.Figure | tuple[plt.Figure, pd.DataFrame]`, so a type checker or IDE sees
  the real contract. Found by the new mypy job.
- Assorted type fixes surfaced by the same job: a missing variable annotation in
  `stats.py`, a widened `GridSpec` annotation, string tick labels where matplotlib
  expects them, `plt.get_cmap("viridis_r")` in place of the attribute access its
  stubs lack, and consistent float figure sizes.
- **`CONTRIBUTING.md` omitted mypy.** The mypy CI job was added without updating
  the "checks that must pass" section, which still said two checks (pytest and
  ruff). A contributor following it would hit a CI failure it never mentioned.
  The `dev` extra description was also stale.
- The contributing guide and the generative AI disclosure no longer hardcode the
  supported Python range, which went stale when 3.14 was added. Both now point at
  the source of truth instead.
- **Subtitles no longer print over the title** in `detection_waterfall()` and
  `missing_runorder()`. The subtitle was placed at a fraction of the axes height
  and the title at a fixed point offset, so the two landed in the same band. Both
  now use point offsets and stack at any figure size.
- **`missing_matrix()` stays readable on tall matrices.** On a matrix of about
  2,000 features the annotation strips and the completeness sparkline had fixed
  heights and shrank to a few pixels, and the rotated sample labels ran into the
  sparkline below the matrix. The strips and sparkline now grow with the matrix
  (small matrices keep their old sizes), the sample labels go on the bottom
  panel, and the dendrogram drops its distance axis, whose ticks collided with the
  first annotation strip.
- **`detection_waterfall()` threshold labels.** Each threshold now drops a dashed
  line to its feature count on the x-axis, and the label reads
  "1,377 proteins at ≥50% detection" without the percentage of the total. The
  droplines are drawn only for the single pooled curve: with `group_level`, each
  group's curve crosses the threshold somewhere else, and the pooled count would
  mark a point on none of them.

## [0.3.1] - 2026-08-11

Documentation release. No functional changes to the library.

### Added

- **`Examples` sections for every public function.** Three of thirteen had one.
  The documentation site renders these docstrings directly, so the gap was visible
  on every API page.
- **`tests/test_docstring_examples.py`** executes every `>>>` line in a public
  docstring against a shared fixture. A documented example that stops working now
  fails the build, rather than sitting in the rendered docs looking plausible.

### Fixed

- **The tutorial notebook did not run.** Its validation section called `qc(df)` and
  `assert_qc(df)`, but the notebook builds its matrix as `prot_subset`; `df` was
  never defined, so executing the notebook end to end died with `NameError`. The
  section had been committed without being executed.
- **The notebook cell labelled "Lenient thresholds: passes" did not pass.** It
  asked for 50% per-sample completeness on a subset selected for high missingness
  variance, where the worst sample sits at 15%. Lowered to 10%, with a comment
  explaining why the bar is low. The notebook now executes cleanly against the real
  CPTAC data: 21 of 21 cells, no errors.
- **"`return_data=True` on every plot function" was inaccurate** in the README,
  `CLAUDE.md`, the API reference and the generative AI disclosure. Six of the nine
  plot functions accept it. `missing_mechanism()` always returns
  `(Figure, DataFrame)` and needs no flag; `missing_abundance_density()` returns a
  figure and `missing_matrix_html()` returns an HTML string, so neither has a
  tabular result to hand back. Corrected in all four places.

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
