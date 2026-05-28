# Plan: Validation Layer, Interop, and CLI Scope

Date: 2026-05-28
Status: Drafted, not started
Target: Mid-July 2026 resubmission to pyOpenSci

---

## Context

pyOpenSci editor Kylen classified `mismap-qc` as a **data validation** package, not a visualization one (issue [pyopensci/software-submission#293](https://github.com/pyOpenSci/software-submission/issues/293)). The current API is plot-heavy. To make the validation framing real and survive reviewer scrutiny, the package needs:

1. A programmatic validation API that returns structured results, not just figures.
2. Numeric outputs alongside every plot, so users can act on the data.
3. Ecosystem interop, so the package reads what omics users actually have.
4. A CLI, so it drops into pipelines without Python boilerplate.
5. A selective set of adjacent QC checks that stay within the missing-data niche.

This document is the companion to `PLAN_new_plots.md`. Plots there stay; this adds a parallel track.

---

## Scope summary

| Scope | Items | Effort | Strategic value |
|---|---|---|---|
| A | `MismapReport`, `qc()`, `assert_qc()` | medium | very high |
| B | `return_data=True` on every plot | low | high |
| C | `from_anndata`, MaxQuant / DIA-NN / FragPipe readers | medium | high |
| D | `mismap-qc` CLI | small-medium | medium |
| E | LOD estimation, imputation diagnostic, replicate concordance | medium | medium |

Wave 1 / Wave 2 plot work from `PLAN_new_plots.md` continues in parallel.

---

# Scope A — Programmatic Validation Layer

The load-bearing piece. Without this, the package looks like a viz library that called itself a validator.

## A.1 — `MismapReport` object

### What it does

Single dataclass-style object that holds the full QC result for a dataset. Returned by `qc()` and consumed by `assert_qc()`. Has serialization helpers so a CI pipeline can dump it to JSON, paste it into a Slack message, or render an HTML summary without re-running anything.

### Why it matters

Right now every QC question requires calling a plot function, looking at it, and writing a follow-up script. A single object that holds all the answers turns mismap-qc from "an inspector" into "a check." This is the single biggest signal to pyOpenSci reviewers that this is a validation package.

### API sketch

```python
from dataclasses import dataclass
import pandas as pd


class MismapQCWarning(UserWarning):
    """Emitted when a warning-severity rule is violated."""


@dataclass(frozen=True)
class RuleResult:
    rule: str                 # e.g. "min_sample_completeness"
    severity: str             # "error" | "warning" | "info"
    passed: bool
    threshold: float
    actual: float
    detail: str = ""          # human-readable specifics (offending samples, etc.)


@dataclass(frozen=True)
class MismapReport:
    # Inputs (for reproducibility)
    n_features: int
    n_samples: int
    group_level: str | int | None

    # Per-sample
    sample_missing_rate: pd.Series          # index: sample, value: % missing
    sample_outliers: pd.DataFrame           # cols: sample, group, z_score, flagged

    # Per-feature
    feature_detection_rate: pd.Series       # index: feature, value: % detected
    feature_mechanism: pd.DataFrame | None  # cols: feature, mechanism, p_value, ...
    feature_codropouts: pd.DataFrame | None # top co-missing pairs

    # Per-group
    group_completeness: pd.Series | None    # index: group, value: mean detection rate

    # Batch / temporal (optional)
    batch_test: pd.DataFrame | None         # volcano-table from batch_missing_test()
    runorder_trend: dict | None             # {"slope": float, "p": float, ...}

    # Threshold evaluation results
    results: tuple[RuleResult, ...] = ()

    # Computed properties (filter `results` by severity + pass status)
    @property
    def errors(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.results if r.severity == "error" and not r.passed)

    @property
    def warnings(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.results if r.severity == "warning" and not r.passed)

    @property
    def info(self) -> tuple[RuleResult, ...]:
        return tuple(r for r in self.results if r.severity == "info" and not r.passed)

    @property
    def passed(self) -> bool:
        return not self.errors

    # No-raise threshold evaluation against an existing report
    def check(
        self,
        thresholds: dict,
        severity_overrides: dict | None = None,
    ) -> tuple[RuleResult, ...]:
        """Re-evaluate thresholds; return RuleResult tuple. Does not raise, does not emit warnings."""
        ...

    def passes(self, thresholds: dict, **kw) -> bool:
        """Sugar: True iff no error-severity rule would fail at these thresholds."""
        return not any(r.severity == "error" and not r.passed for r in self.check(thresholds, **kw))

    # Serialization
    def to_dict(self) -> dict: ...
    def to_json(self, path: str | None = None) -> str: ...
    def to_html(self, path: str | None = None) -> str: ...
    def __repr__(self) -> str: ...   # one-line summary
    def summary(self) -> str: ...    # multi-line readable summary
```

### Behavior details

- Frozen dataclass: `MismapReport` instances are immutable. Re-run `qc()` for a fresh analysis.
- `__repr__` returns one line, e.g. `MismapReport(n=8412x96, 3 outliers, 412 MNAR features, passed=True)`.
- `summary()` returns a multi-line human-readable string with section headers. Used by `print(report)` via `__str__`.
- `passed` is a computed property: `True` iff no error-severity rule failed. Warnings do not block.
- `errors`, `warnings`, `info` properties filter `results` by severity and pass status. They are tuples, not lists, to preserve immutability.
- `check(thresholds)` re-evaluates rules against an existing report without re-running the analysis. Returns a tuple of `RuleResult` without raising and without emitting warnings. Useful for "what would happen if I tightened this rule?" exploration in notebooks.
- `passes(thresholds)` returns a bool. Sugar for the common "did it pass?" check.
- `to_dict()` produces a JSON-serializable dict. DataFrames go through `df.to_dict(orient="records")`; `RuleResult` instances via `dataclasses.asdict()`.
- `to_json()` writes to disk if path given, returns string either way. Uses `default=str` to handle numpy types.
- `to_html()` produces a self-contained HTML fragment (no `<html>` wrapper) suitable for embedding. Full report generation lives in `missing_summary_report()` from the existing plan.
- Optional fields default to `None`. The presence of `batch_test` or `feature_mechanism` indicates the user requested those checks.

### Implementation notes

- Use `dataclasses.dataclass`, not Pydantic. Avoid adding a heavy dependency.
- Numpy / pandas types in `to_dict` need explicit conversion. Write a private `_jsonify(obj)` helper.
- No methods that re-run analysis. `MismapReport` is a passive container.

### Tests

- Construct an empty report, assert defaults.
- Round-trip: `MismapReport(...).to_dict()` then back via `MismapReport(**d)`.
- `to_json` with and without path.
- `to_html` produces a string containing expected section headers.
- `__repr__` is single-line and contains key counts.

---

## A.2 — `qc()` entry point

### What it does

Single function that runs every check on a dataset and returns a `MismapReport`. Users who want "just tell me if this is okay" call this and inspect the result.

### Why it matters

A pyOpenSci reviewer should be able to read the README, see one example, and understand the package. `qc(df)` is that example.

### API sketch

```python
def qc(
    df: pd.DataFrame,
    *,
    group_level: int | str | None = None,
    run_order: pd.Series | list | None = None,
    checks: tuple[str, ...] = ("completeness", "outliers", "mechanism", "codropouts"),
    thresholds: dict | None = None,
    severity_overrides: dict | None = None,
    feature_type: str = "PROT",
    verbose: bool = False,
) -> MismapReport:
```

### Behavior details

- `checks` selects which analyses run. Default is the "standard battery":
  - `"completeness"` -> per-sample, per-feature, per-group rates.
  - `"outliers"` -> calls the logic from `sample_outlier_score()`.
  - `"mechanism"` -> calls `missing_mechanism()`.
  - `"codropouts"` -> top-N co-missing feature pairs.
  - `"batch"` -> calls `batch_missing_test()`, requires `group_level` with exactly 2 levels.
  - `"runorder"` -> requires `run_order`, computes trend statistic.
- `thresholds` is the same dict format consumed by `assert_qc()` (see A.4 vocabulary). If provided, `qc()` evaluates each rule and populates `report.results`. Warning-severity violations emit `MismapQCWarning` via the `warnings` module so they surface by default but can be silenced with `warnings.filterwarnings("ignore", category=MismapQCWarning)`. If `thresholds` is not provided, no rules are evaluated and `report.passed` is `True` by definition.
- `severity_overrides={"rule_name": "warning" | "error" | "info"}` lets a lab soften or stiffen default severities at call time without subclassing or monkey-patching. Unknown rule names raise `ValueError`.
- `verbose=True` prints a progress line per check (useful for large datasets where mechanism testing is slow).
- Returns a fully populated `MismapReport`. Skipped checks leave the corresponding field `None`.

### Implementation notes

- Internally, `qc()` is glue. It calls existing plot functions in their `return_data=True` mode (see Scope B) and packages the resulting DataFrames into the report.
- Cache nothing. Each call is independent.
- Long-running checks (mechanism on >50k features) should print a warning when `verbose=False` so a stalled call is not silent.

### Tests

- Default checks on synthetic data return a `MismapReport` with all expected fields populated.
- `checks=("completeness",)` populates only completeness fields, rest are `None`.
- `thresholds={"min_sample_completeness": 0.99}` on data with known low-completeness samples produces `passed=False` and a populated `failures` list.
- `group_level` and `run_order` paths exercised.
- Real-data test (CPTAC) returns a report with sensible numbers.

---

## A.3 — `assert_qc()` guard

### What it does

Thin wrapper around `qc()` that raises `MismapQCFailure` if any threshold rule fails. Designed to drop into a pipeline, a notebook cell, or a CI job.

### Why it matters

This is what makes mismap-qc a *validator* rather than an inspector. The pattern is intentionally familiar to anyone who has used `assert` in tests or `great-expectations` for tabular data.

### API sketch

```python
class MismapQCFailure(AssertionError):
    """Raised by assert_qc when one or more error-severity QC rules fail."""
    def __init__(self, report: "MismapReport"):
        self.report = report
        super().__init__(self._format())
    def _format(self) -> str:
        """Render report.errors (and any suppressed warnings) into the block shown below."""
        ...


def assert_qc(
    df: pd.DataFrame,
    *,
    thresholds: dict,
    severity_overrides: dict | None = None,
    **qc_kwargs,
) -> MismapReport:
    report = qc(
        df,
        thresholds=thresholds,
        severity_overrides=severity_overrides,
        **qc_kwargs,
    )
    if not report.passed:
        raise MismapQCFailure(report)
    return report
```

### Behavior details

- `thresholds` keys are rule names (see A.4 vocabulary for the locked v0.2.0 set), values are numeric thresholds.
- `assert_qc` raises only on error-severity failures. Warning-severity violations emit `MismapQCWarning` via the `warnings` module but do not raise. The returned report has both populated in `report.errors` and `report.warnings`.
- Two no-raise alternatives exist as methods on `MismapReport`:
  - `report.check(thresholds=...)` returns a tuple of `RuleResult` without raising. Use after inspecting a report.
  - `report.passes(thresholds=...)` returns a `bool` (True iff no error-severity rule would fail). Sugar.
- `severity_overrides` allows changing default severities at call time without subclassing or monkey-patching.
- Returns the populated report on success, so a user can write `report = assert_qc(df, thresholds=...)` and keep going.

### Implementation notes

- `MismapQCFailure.__str__` lists each failed rule with severity, threshold, actual value, and offenders. Warnings are mentioned as suppressed so the user does not miss them. Example:
  ```
  MismapQCFailure: 2 rules failed (1 warning suppressed)

    [ERROR] min_sample_completeness
      threshold: 0.60
      actual minimum: 0.34
      offenders: S07 (0.34), S22 (0.41), S81 (0.58)

    [ERROR] max_mnar_fraction
      threshold: 0.30
      actual: 0.41 (1652 / 4023 testable features)

    [WARNING] max_runorder_slope  (suppressed, did not raise)
      threshold: 0.005
      actual: 0.012
  ```
- Document the exception class in the public API so users can catch it specifically.

### Tests

- Passing thresholds returns a report.
- Failing threshold raises `MismapQCFailure`.
- Failure message contains rule names and actual values.
- Unknown threshold key raises `ValueError` at parse time, not silent skip.

---

## A.4 — Threshold rules vocabulary

### Locked rules for v0.2.0

Every rule maps directly to a check that `qc()` already runs. No rule introduces new analysis.

| Rule | Type | From check | Default severity | Notes |
|---|---|---|---|---|
| `min_sample_completeness` | float | completeness | error | Worst sample's detection rate must be at least this |
| `min_sample_completeness_per_group` | float | completeness | error | Per-group variant. Worst sample in every group |
| `min_features_detected` | int | completeness | error | Absolute count of features detected anywhere |
| `max_feature_missing_rate` | float | completeness | warning | No feature missing in more than this fraction of samples |
| `max_sample_outliers` | int | outliers | error | Count of samples flagged as outliers |
| `max_sample_outlier_zscore` | float | outliers | error | Largest absolute z-score across samples |
| `max_mnar_fraction` | float | mechanism | warning | Fraction of testable features classified MNAR |
| `max_unclassified_fraction` | float | mechanism | info | Features with insufficient data for classification |
| `min_group_completeness` | float | completeness | error | Per-group mean detection rate (every group must meet) |
| `max_batch_effect_features` | int | batch | warning | Significantly enriched features. Requires `group_level` with 2 levels |
| `max_runorder_slope` | float | runorder | warning | Linear slope of missingness vs run order. Requires `run_order` |

### Naming convention

- `min_X` / `max_X` prefixes indicate which direction is bad.
- `_per_group` suffix means the rule applies within each group rather than globally. Default is global.
- Singular subject: `min_sample_completeness` checks the worst sample; `min_group_completeness` checks the worst group.

### Severity defaults (rationale)

| Severity | Used for |
|---|---|
| `error` | Rules that would invalidate downstream analysis if violated (sample completeness, outliers, group completeness, hard count floors) |
| `warning` | Rules that suggest a problem but do not block analysis (MNAR fraction, batch effects, runorder drift, per-feature missingness) |
| `info` | Rules that surface useful context but rarely justify any action (unclassified fraction) |

Defaults are conservative. Users override with `severity_overrides` when a lab's tolerances differ.

### Explicit non-rules for v0.2.0

These will be asked for. Document why they are deferred so reviewers and users do not assume they were forgotten:

| Non-rule | Why deferred |
|---|---|
| `min_replicate_concordance` | Depends on `replicate_concordance()` (Scope E.3, post-resubmission) |
| `max_codropout_fraction` | Depends on `comissing_heatmap` output schema being stable |
| `min_protein_count_with_imputation` | Implies imputation, which mismap-qc explicitly does not do |
| User-defined custom rules | Deferred to v0.3 plugin system. Workaround: evaluate manually and inspect `report.results` directly |

### Adding rules later

The rule registry is internal but versioned. Adding a rule is non-breaking. Renaming, removing, or changing the default severity of an existing rule requires a deprecation cycle (one minor version with a `DeprecationWarning`).

---

# Scope B — Numeric Outputs Alongside Plots

### What it does

Every plot function gains a `return_data: bool = False` parameter. When `True`, the function returns `(fig, data_df)` instead of just `fig`. The DataFrame schema is documented in the docstring.

### Why it matters

Reviewers should never have to ask "and how do I get the numbers behind this?" The answer is the same flag on every function.

### Behavior details

Each function returns a DataFrame with a stable schema:

| Function | Returned columns |
|---|---|
| `missing_matrix` | `feature, sample, missing` (long form) |
| `completeness_bars` | `group, completeness, n_samples` |
| `detection_waterfall` | `feature, detection_rate, rank` |
| `missing_runorder` | `sample, run_order, missing_rate, group` |
| `missing_mechanism` | `feature, mechanism, missing_rate, mean_abundance, p_value` (already returns this; keep the same schema) |
| `comissing_heatmap` | `feature_a, feature_b, comissingness` (long form, only pairs above a min) |
| `missing_upset` | `intersection, size, members` |
| `sample_outlier_score` | `sample, group, missing_rate, z_score, flagged` (already returns this) |
| `batch_missing_test` | `feature, log2_OR, p_value, q_value, enriched_in, significant` (already returns this) |

Functions that *already* return a DataFrame (mechanism, outlier_score, batch_test) skip the `return_data` flag and always return the tuple. This is a small backwards-compatibility break; flag it in the changelog.

### Implementation notes

- Compute the data once internally, plot from it, and return both. Do not re-compute.
- Document the schema in the docstring under a `Returns` section using NumPy docstring style.
- All sample / feature columns use strings, never numeric indices.

### Tests

- For each plot function: call with `return_data=True`, assert tuple length 2, assert DataFrame columns match the schema above.
- Schema regression test: a single `test_return_data_schemas.py` that imports a registry dict mapping function -> expected columns, and asserts each one.

---

# Scope C — IO and Ecosystem Interop

Three sub-items, ranked by strategic value. Implement C.1 before mid-July; defer the others until post-review.

## C.1 — `from_anndata()` reader

### What it does

Converts an `AnnData` object into the features × samples DataFrame the rest of the API expects. Optionally pulls `obs` columns through as MultiIndex levels.

### Why it matters

scverse is the dominant ecosystem for single-cell and increasingly bulk omics in Python. AnnData support means single-cell proteomics users (who have severe missingness problems) can use the package without reformatting. AnnData also has a pyOpenSci-affiliated maintainer community, which improves review odds.

### API sketch

```python
def from_anndata(
    adata,                               # AnnData
    *,
    layer: str | None = None,            # which layer to use; None = .X
    missing_value: float | str = "nan",  # "nan", "zero", or float threshold
    obs_levels: list[str] | None = None, # obs columns -> MultiIndex levels
    var_index: str | None = None,        # var column to use as feature names; None = adata.var_names
    transpose: bool = True,              # AnnData is obs (samples) x var (features); we want the inverse
) -> pd.DataFrame:
```

### Behavior details

- Default produces features × samples. Set `transpose=False` if the user's AnnData is already in features × samples form (rare but possible).
- `missing_value="nan"` keeps NaN as missing (the common case).
- `missing_value="zero"` converts exact zeros to NaN. Useful for proteomics intensities stored as zero rather than NaN.
- `missing_value=<float>` treats values below the threshold as missing. Useful for log-intensity matrices with a noise floor.
- `obs_levels=["batch", "condition"]` produces a 2-level MultiIndex on columns, ready for `group_level="condition"`.
- Sparse layers are densified with a warning if the result exceeds 1 GB.

### Implementation notes

- AnnData is an optional dependency: `pip install mismap-qc[anndata]`. Import inside the function with a clear error.
- Do not import `scanpy`. AnnData alone is much lighter.
- Validate that requested `obs_levels` exist before any conversion to avoid wasted work on big objects.

### Tests

- Synthetic `AnnData` with NaN -> round-trip to DataFrame, then back through `missing_matrix()`.
- `missing_value="zero"` converts zeros correctly.
- `obs_levels` produces correct MultiIndex.
- Missing AnnData dependency raises a clear `ImportError` with install hint.

## C.2 — Proteomics search-engine output parsers (post-resubmission)

### What it does

Direct readers for the file formats users actually have:

```python
from_maxquant("proteinGroups.txt")         # MaxQuant
from_diann("report.tsv")                    # DIA-NN main output
from_fragpipe("combined_protein.tsv")       # FragPipe
from_spectronaut("Report.tsv")              # Spectronaut
```

Each returns a features × samples DataFrame with the sample names parsed out and an optional `metadata` DataFrame for sample annotations.

### Why it matters

This is the path to actual community adoption. A proteomics user does not currently see how to get from `proteinGroups.txt` to mismap-qc input. Adding readers closes that gap.

### Implementation notes

- Each parser owns the format quirks (contaminant rows in MaxQuant, decoy/REV in DIA-NN, intensity vs LFQ vs iBAQ column selection, etc.).
- Parameter `intensity: str` selects which intensity column family to use. Default per format documented.
- Each parser is a separate ~50-80 line function. They share zero code; keep them dumb and explicit.
- Defer until post-review. The package can ship without these and reviewers will accept "AnnData is the supported entry point" as long as that path works.

### Tests

- Small fixture files for each format (anonymized samples, ~5 features × 3 samples) committed to `tests/fixtures/io/`.
- One test per parser: load the fixture, assert shape and column types.
- Real-data smoke test (skipped if files absent) for each format.

## C.3 — RNA-Seq readers (post-resubmission)

Lower priority because RNA-Seq users typically already have a count matrix in DataFrame form, and the standard formats (featureCounts, salmon, STAR) are already covered by `pandas.read_csv` with the right flags. A thin wrapper is mostly a UX nicety. Defer indefinitely unless a user requests it.

---

# Scope D — CLI

### What it does

A `mismap-qc` command that takes a TSV/CSV/AnnData file and produces a report or specific plot from the shell.

### Why it matters

Two reasons. First, CI pipelines run shell commands, not Python imports. A CLI lets users wire mismap-qc into Nextflow / Snakemake / GitLab CI directly. Second, it is a maturity signal pyOpenSci reviewers consistently note.

### API sketch

```
mismap-qc qc DATA.tsv --groups META.tsv --thresholds rules.yml
mismap-qc plot matrix DATA.tsv --out matrix.png
mismap-qc plot waterfall DATA.tsv --groups META.tsv --out waterfall.png
mismap-qc report DATA.tsv --groups META.tsv --out report.html
```

Subcommands:
- `qc` runs `assert_qc()` with rules loaded from a YAML file. Exits non-zero on failure. Prints a colored summary to stdout.
- `plot <name>` runs one plot function and saves the figure.
- `report` runs `missing_summary_report()` (from Wave 2).

### Behavior details

- Input format auto-detected from extension: `.tsv`, `.csv`, `.h5ad`.
- `--groups META.tsv` is a separate metadata file with at minimum a `sample` column. Any additional columns become MultiIndex levels.
- `--thresholds rules.yml` loads a YAML mapping of rule name to threshold. Example:
  ```yaml
  min_sample_completeness: 0.60
  max_sample_outliers: 3
  max_mnar_fraction: 0.30
  ```
- `qc` exit codes: 0 on pass, 1 on QC failure, 2 on usage error, 3 on data error.

### Implementation notes

- Use `typer` (cleaner than `click` for this scale, and pyOpenSci packages often standardize on it).
- Entry point declared in `pyproject.toml`:
  ```toml
  [project.scripts]
  mismap-qc = "mismap_qc.cli:app"
  ```
- This means the CLI code needs to move out of the single-file constraint a bit. Suggested split: keep `mismap_qc.py` as the library, add `mismap_qc/cli.py` only when implementing this. Re-check the single-file convention with the user before doing so.
- Defer until A and B are done. Without `MismapReport` and structured returns, the CLI has nothing clean to wrap.

### Tests

- `typer.testing.CliRunner` for each subcommand.
- `qc` with passing rules: exit 0.
- `qc` with failing rules: exit 1, stderr contains failure summary.
- `plot matrix` with a temp output path: file exists, is a valid PNG.

---

# Scope E — Domain-specific QC Checks

Three additions that stay strictly inside missing-data territory. Each is independently useful and can be added incrementally. Pick at most one for the mid-July push.

## E.1 — `estimate_lod()`

### What it does

Estimates a per-feature limit of detection from observed values. For each feature, the LOD is taken as the lowest observed value across samples, or the 5th percentile of observed values, depending on method. Returns a Series indexed by feature.

### Why it matters

Many proteomics users distinguish "missing because below LOD" (MNAR) from "missing for other reasons" (MAR). An explicit LOD estimate makes that distinction quantitative and feeds into `missing_mechanism()`.

### API sketch

```python
def estimate_lod(
    df: pd.DataFrame,
    *,
    method: str = "min",          # "min" or "quantile"
    quantile: float = 0.05,        # used when method="quantile"
    min_present: int = 3,
) -> pd.Series:
```

### Implementation notes

- `method="min"`: per-feature minimum of non-NaN values.
- `method="quantile"`: per-feature quantile of non-NaN values.
- Features with fewer than `min_present` observations get `NaN` LOD (cannot estimate).
- Returns a Series; no figure. This is a pure analysis function.

### Tests

- Synthetic data with known minimums: assert recovered.
- All-missing feature returns NaN.
- `min_present` threshold respected.

## E.2 — `imputation_diagnostic()`

### What it does

Takes a pre-imputation DataFrame and a post-imputation DataFrame, produces a diagnostic figure showing:
1. Intensity distribution overlay (KDE) before vs after.
2. Per-feature change in mean intensity (scatter, only previously missing positions).
3. KS-test statistic comparing the imputed-value distribution to the observed distribution at the low-intensity tail.

### Why it matters

The package does not implement imputation (good, that is a different scope), but it does help users *evaluate* whether their imputation reasonable. Distinguishes a downshift imputation that respects MNAR structure from one that distorts the distribution.

### API sketch

```python
def imputation_diagnostic(
    df_pre: pd.DataFrame,
    df_imputed: pd.DataFrame,
    *,
    feature_type: str = "PROT",
    title: str | None = None,
    save: str | None = None,
    dpi: int = 150,
    return_data: bool = False,
) -> plt.Figure | tuple[plt.Figure, dict]:
```

### Implementation notes

- Validate that `df_pre` and `df_imputed` have identical shape and index.
- "Imputed positions" are those where `df_pre.isna() & df_imputed.notna()`.
- KS-test via `scipy.stats.kstest` or `ks_2samp` between the imputed-value distribution and the lowest decile of observed values.

### Tests

- Identity case: `df_imputed == df_pre` (no imputation) produces empty imputed-position set, function returns gracefully with warning.
- Synthetic downshift imputation: distributions differ only at low tail, KS p-value low.
- Synthetic mean imputation: distributions differ across full range, distinct from MNAR pattern.

## E.3 — `replicate_concordance()`

### What it does

For each group, computes pairwise Jaccard similarity of the set of detected features across replicates. Produces a heatmap of pairwise concordance and a bar chart of mean concordance per group.

### Why it matters

Replicates that detect very different feature sets indicate sample handling issues. Within-group concordance is a sensitive QC metric that current visualizations do not surface directly.

### API sketch

```python
def replicate_concordance(
    df: pd.DataFrame,
    group_level: int | str,
    *,
    metric: str = "jaccard",     # "jaccard" or "overlap"
    feature_type: str = "PROT",
    title: str | None = None,
    save: str | None = None,
    dpi: int = 150,
    return_data: bool = False,
) -> plt.Figure | tuple[plt.Figure, pd.DataFrame]:
```

### Implementation notes

- For each pair of samples within a group: compute `|A ∩ B| / |A ∪ B|` where A and B are sets of detected features.
- Layout: heatmap of pairwise concordance with samples ordered by group, group blocks visible. Side bar chart of group means.
- `metric="overlap"` uses `|A ∩ B| / min(|A|, |B|)` instead (Szymkiewicz-Simpson).

### Tests

- Identical replicates: concordance 1.0.
- Disjoint replicates: concordance 0.0.
- Group-level mean computed correctly.

---

# Cross-cutting concerns

## Testing

Each scope item follows the same pattern already established in `tests/test_mismap_qc.py`:

1. Returns expected type (figure, dataclass, Series, etc.)
2. Works with flat columns and MultiIndex.
3. Handles all-missing and all-present edge cases without crashing.
4. `save` / output-path arguments produce a file.
5. Real-data smoke test in `tests/test_*_realdata.py`, skipped if CPTAC data absent.

New cross-cutting tests:

- `tests/test_report_serialization.py` — round-trip `MismapReport` through `to_dict`, `to_json`, `to_html`.
- `tests/test_return_data_schemas.py` — registry-driven test that every plot function returns the documented schema.
- `tests/test_cli.py` — `typer.testing.CliRunner` per subcommand (only if D is implemented).

## Documentation

For the mid-July resubmission, the README needs a new section that demonstrates the validation framing:

```python
import pandas as pd
from mismap_qc import qc, assert_qc

df = pd.read_csv("proteomics.tsv", sep="\t", index_col=0)

# Inspect
report = qc(df)
print(report)
# MismapReport(n=8412x96, 3 outliers, 412 MNAR features, passed=True)

# Or guard a pipeline
assert_qc(df, thresholds={
    "min_sample_completeness": 0.60,
    "max_mnar_fraction": 0.30,
})
```

This single example carries more weight with reviewers than any individual plot screenshot.

## Single-file convention

Scopes A, B, E stay inside `mismap_qc.py`. Scope C.1 (`from_anndata`) also fits. Scope D (CLI) requires a second file (`cli.py`). The existing CLAUDE.md says "Don't split into submodules unless it exceeds ~3,000 lines." The CLI is the natural exception, since it lives behind a separate entry point. Flag this to the user before implementing D.

---

# Implementation order

Time budget is variable, so the plan is structured as checkpoints rather than a
weekly schedule. Each checkpoint is a self-contained shippable state. You can
stop at any one and have made the package meaningfully more reviewable for
pyOpenSci.

| # | Checkpoint | Done when | Resubmission viable? |
|---|---|---|---|
| 0 | Today | Wave 1 has 4/5 plots, no validation API | No |
| 1 | Wave 1 closed | `missing_mechanism` and `comissing_heatmap` shipped, tested, in CPTAC notebook | Marginal. More plots, but no structural change |
| 2 | Validation API live | `MismapReport`, `qc()`, `assert_qc()` working end-to-end on CPTAC; tests for all three | Yes. Minimum credible resubmit |
| 3 | Data flow consistent | `return_data=True` on every plot, schema regression test passing | Yes, stronger |
| 4 | Interop opened | `from_anndata` working, README rewritten around the validation framing | Yes, much stronger |
| 5 | Adjacent QC | One Scope E item shipped (recommend `estimate_lod`), CHANGELOG, 0.2.0 tag | Yes, ideal |

**Checkpoint 2 is the real bar.** Everything before it is more of the same;
everything from 2 onward is the package changing shape. If only checkpoint 2
is reached, resubmit anyway. The validation framing is what Kylen will evaluate.

Soft date anchors (use as drift signals, not commitments):
- Hitting checkpoint 4 by mid-July keeps the originally drafted resubmission date.
- Hitting checkpoint 2 by early August still hits the 6-month window from first commit (2026-03-11).
- Past September the package crosses the comfortable 6-month bar regardless of progress; resubmit with whatever is done.

## Post-checkpoint-5 (after resubmission)

- Scope C.2 (search-engine parsers for MaxQuant, DIA-NN, FragPipe, Spectronaut)
- Scope D (CLI)
- Remaining Scope E items (`imputation_diagnostic`, `replicate_concordance`)
- Wave 2 plots (`missing_upset`, `sample_outlier_score`, `batch_missing_test`, `missing_summary_report`) from `PLAN_new_plots.md`

---

# Risks and explicit non-goals

## Risks

- **Scope ballooning**. Each of these scopes contains 2-5 sub-items. The mid-July push only needs A, B, C.1, and one E item to make a credible "iterative maturity" case. Resist the urge to ship everything.
- **`MismapReport` design churn**. Get the fields right the first time; adding a field is easy, removing one breaks downstream code. Lock the schema before publishing 0.2.0.
- **CLI complexity drift**. The CLI must stay a thin wrapper. If a flag does not map to a library function call, the library is missing something and the flag should not exist.

## Non-goals

The following are explicitly out of scope for this package, and should be stated so in the README:

- **Imputation methods**. mismap-qc diagnoses missingness; it does not impute. `imputation_diagnostic()` evaluates external imputation, but the package never imputes.
- **Differential abundance**. `batch_missing_test()` tests *missingness* differences, not abundance differences. The latter is what limma / DESeq2 / MSstats are for.
- **Normalization**. mismap-qc assumes a normalized matrix as input. Normalization belongs upstream.
- **Single-cell zero-inflation modeling**. The statistical framing for single-cell (zero-inflated negative binomial, dropout vs true zero) is distinct from bulk missingness. mismap-qc supports single-cell *data* via AnnData input, but does not provide single-cell-specific tests.

Stating these non-goals in the README is itself a pyOpenSci review signal. Reviewers reward packages that know what they are not.
