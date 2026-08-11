# mismap-qc

Missing-data validation for proteomics and RNA-Seq experiments.

It detects outlier samples, classifies dropout mechanism (MNAR versus MAR), tests for batch effects, and gates pipelines on configurable QC rules. Every check has a matching plot for when you want to see the problem rather than just check it.

## Install

```bash
pip install mismap-qc
```

Optional extras: `[interactive]` for HTML matrices (plotly), `[anndata]` for AnnData input, `[upset]` for `missing_upset()`.

## What it validates

| Check | What it catches |
|---|---|
| Sample completeness | Samples with too few detected features |
| Outlier detection | Samples with anomalous missingness against their group peers |
| Missingness mechanism | Dropouts driven by low abundance (MNAR) against random (MAR) |
| Batch effects | Features whose detection differs between conditions |
| Run order drift | Instrument degradation over a long acquisition |

## The shape of the API

Three entry points, in increasing order of strictness:

```python
from mismap_qc import qc, assert_qc

report = qc(df, group_level="condition")     # always returns a MismapReport
report.passes(thresholds={...})               # bool, raises nothing
assert_qc(df, thresholds={...})               # raises MismapQCFailure
```

`qc()` returns a frozen [`MismapReport`](api/validation.md) holding pandas DataFrames you can drill into. `assert_qc()` is the pipeline gate.

See [Quickstart](quickstart.md) for a worked example, or the [Tutorial](tutorial.md) for real CPTAC proteomics data.

## Why this instead of something else

- `missingno` does general missing-data visualization with no omics awareness: no groups, no MultiIndex sample annotations, no MNAR mechanism.
- `protti` (R) classifies missingness mechanism but has no Python equivalent.
- `great-expectations` validates tabular data but does not understand missingness mechanism or omics-specific patterns.

mismap-qc covers all three through one API and reads AnnData natively.

## Input format

A pandas DataFrame with features as rows, samples as columns, and `NaN` meaning missing or not detected. When the columns are a `MultiIndex`, the level names become annotation strip labels automatically.
