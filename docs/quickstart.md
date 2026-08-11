# Quickstart

## Inspect a dataset

```python
import pandas as pd
from mismap_qc import qc

df = pd.read_csv("proteomics.tsv", sep="\t", index_col=0)

report = qc(df, group_level="condition")
print(report)
# MismapReport(n=8412x96, 3 outliers, 412 MNAR features, passed=True)
```

`report` is a frozen snapshot. Re-run `qc()` to get a new one; it never mutates.

For a readable multi-section view rather than the one-line repr:

```python
print(report.summary())
```

## Drill into what was flagged

The report holds pandas DataFrames, so anything flagged can be queried directly:

```python
report.sample_outliers.query("flagged")
report.feature_mechanism.query("mechanism == 'MNAR'")
```

## Gate a pipeline

`assert_qc()` raises `MismapQCFailure` when a rule is violated:

```python
from mismap_qc import assert_qc

assert_qc(df, thresholds={
    "min_sample_completeness": 0.60,
    "max_mnar_fraction": 0.30,
    "max_sample_outliers": 3,
})
```

Rules carry a default severity of error, warning, or info. Warning-severity violations emit `MismapQCWarning` through the standard `warnings` module rather than raising, and can be silenced the usual way:

```python
import warnings
from mismap_qc import MismapQCWarning

warnings.filterwarnings("ignore", category=MismapQCWarning)
```

Override a rule's severity per call with `severity_overrides`.

## Get the numbers behind any plot

Every plot function accepts `return_data=True` and returns `(Figure, DataFrame)` against a documented schema:

```python
from mismap_qc import detection_waterfall

fig, table = detection_waterfall(df, return_data=True)
# table columns: feature, detection_rate, rank
```

This matters when a plot truncates. `missing_upset()` caps the figure at the 50 largest intersections, but the returned table contains every intersection with a `plotted` column recording which ones made the figure, so nothing is hidden:

```python
from mismap_qc import missing_upset

fig, table = missing_upset(df, return_data=True)
table.query("~plotted")                       # what the figure left out
table.query("members == 'S1|S3'").feature     # features missing in exactly S1 and S3
```

## Start from AnnData

```python
from mismap_qc import from_anndata, qc

df = from_anndata(adata, obs_levels=["batch", "condition"])
report = qc(df, group_level="condition")
```

`from_anndata()` needs the optional `anndata` extra: `pip install mismap-qc[anndata]`.
