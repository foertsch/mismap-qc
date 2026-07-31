# mismap-qc

[![PyPI version](https://img.shields.io/pypi/v/mismap-qc.svg)](https://pypi.org/project/mismap-qc/)
[![Python versions](https://img.shields.io/pypi/pyversions/mismap-qc.svg)](https://pypi.org/project/mismap-qc/)
[![License: MIT](https://img.shields.io/pypi/l/mismap-qc.svg)](LICENSE)
[![Tests](https://github.com/foertsch/mismap-qc/actions/workflows/tests.yml/badge.svg)](https://github.com/foertsch/mismap-qc/actions/workflows/tests.yml)
[![Views](https://hits.sh/github.com/foertsch/mismap-qc.svg?label=views)](https://hits.sh/github.com/foertsch/mismap-qc/)

Missing-data validation for proteomics and RNA-Seq experiments. Detects outlier
samples, classifies dropout mechanism (MNAR vs MAR), tests for batch effects,
and gates pipelines on configurable QC rules. Every check has a matching plot
for when you want to see the problem rather than just check it.

## Install

```bash
pip install mismap-qc
```

Optional extras: `[interactive]` for HTML matrices (plotly), `[anndata]` for
AnnData input.

## Quick start

```python
import pandas as pd
from mismap_qc import qc, assert_qc

df = pd.read_csv("proteomics.tsv", sep="\t", index_col=0)

# 1. Inspect: full QC report in one call
report = qc(df, group_level="condition")
print(report)
# MismapReport(n=8412x96, 3 outliers, 412 MNAR features, passed=True)

# 2. Drill in on anything flagged (the report holds pandas DataFrames)
report.sample_outliers.query("flagged")
report.feature_mechanism.query("mechanism == 'MNAR'")

# 3. Gate a pipeline (raises MismapQCFailure on rule violation)
assert_qc(df, thresholds={
    "min_sample_completeness": 0.60,
    "max_mnar_fraction": 0.30,
    "max_sample_outliers": 3,
})
```

For a human-readable multi-section summary instead of the one-line repr:

```python
print(report.summary())
```

Starting from an AnnData object:

```python
from mismap_qc import from_anndata, qc
df = from_anndata(adata, obs_levels=["batch", "condition"])
report = qc(df, group_level="condition")
```

To pair any plot with its underlying numbers, pass `return_data=True`:

```python
from mismap_qc import detection_waterfall
fig, table = detection_waterfall(df, return_data=True)
# table: feature, detection_rate, rank
```

## What this validates

| Check | What it catches |
|---|---|
| Sample completeness | Samples with too few detected features |
| Outlier detection | Samples with anomalous missingness vs group peers |
| Missingness mechanism | Dropouts driven by low abundance (MNAR) vs random (MAR) |
| Batch effects | Features whose detection differs between conditions |
| Run order drift | Instrument degradation over a long acquisition |

![demo](output/demo_full.png)

## Why use this

mismap-qc fills a specific gap in the Python omics ecosystem:

- `missingno` does general missing-data visualization but has no omics awareness
  (groups, MultiIndex sample annotations, MNAR mechanism).
- `protti` (R) classifies missingness mechanism but has no Python equivalent.
- `great-expectations` validates tabular data but does not understand
  missingness mechanism or omics-specific patterns.

mismap-qc handles all three with a single API and reads AnnData natively.

## Examples

- **[CPTAC Lung Adenocarcinoma proteomics](examples/cptac_proteomics.ipynb)** --
  real-world tutorial using public CPTAC LUAD data (~100 tumour/normal samples).
  Shows how missingness clusters by tumour/normal status.

## Running the demo

No virtual environment needed -- the demo uses [PEP 723](https://peps.python.org/pep-0723/) inline script dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv run demo.py
```

## Feature types

The `feature_type` parameter controls labels in axes and tooltips:

| Value | Labels |
|-------|--------|
| `"PROT"` | Protein / Proteins (default) |
| `"GENE"` | Gene / Genes |
| `"PEPTIDE"` | Peptide / Peptides |

## Input format

A pandas DataFrame with:
- **Rows** = features (proteins, genes, peptides)
- **Columns** = samples, optionally as a `MultiIndex` for annotation strips
- **NaN** = missing / not detected

When columns are a MultiIndex, level names automatically become annotation strip labels.

## `missing_matrix()` -- static plot

```python
fig = missing_matrix(
    df,
    title="Gene Detection Matrix",
    subtitle="80 genes x 30 samples | 23% missing",
    save="output.png",
)
```

### Layout (top to bottom)

| Component | Description |
|---|---|
| Title + subtitle | Bold title, italic subtitle for metadata |
| Dendrogram | Hierarchical clustering of samples by nullity pattern |
| Annotation strips | One colour bar per MultiIndex column level |
| Nullity matrix | Dark = detected, light = missing |
| Completeness sparkline | Per-sample or per-feature detection rate |

### Parameters

#### Data & labels

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `DataFrame` | required | Features (rows) x samples (columns). NaN = missing. |
| `title` | `str` | `""` | Bold figure title |
| `subtitle` | `str` | `""` | Italic line below title (e.g. dataset metadata) |
| `feature_type` | `str` | `"PROT"` | Feature type: `"PROT"`, `"GENE"`, or `"PEPTIDE"` |
| `label_level` | `int` | `-1` | Which column level to use for x-axis tick labels |

#### Clustering & sorting

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cluster_samples` | `bool` | `True` | Cluster samples by binary nullity pattern |
| `cluster_method` | `str` | `"average"` | scipy linkage method |
| `show_dendrogram` | `bool` | `True` | Show dendrogram above the matrix |
| `sort_features` | `str \| None` | `"descending"` | Sort features by completeness (`"ascending"`, `"descending"`, or `None`) |

#### Annotations

| Parameter | Type | Default | Description |
|---|---|---|---|
| `annotation_levels` | `list[int] \| None` | `None` | Column levels to show as colour bars (default: all except innermost) |
| `annotation_colors` | `dict \| None` | `None` | Custom colours per level (see below) |

Custom annotation colours accept level indices or names as keys:

```python
missing_matrix(
    df,
    annotation_colors={
        "Medium_Type": {"Fresh": "#88CCEE", "Conditioned": "#CC6677"},
        "Medium_Condition": {"SF": "#44AA99", "FBS": "#DDCC77", "AS": "#AA4499"},
    },
)
```

Unspecified factor levels fall back to built-in palettes.

#### Completeness sparkline

| Parameter | Type | Default | Description |
|---|---|---|---|
| `completeness` | `str` | `"below"` | `"below"` = per-sample (horizontal), `"side"` = per-feature (vertical) |
| `completeness_threshold` | `float \| None` | `None` | Draws a dashed red line at this value (0--1) |

#### Legends & layout

| Parameter | Type | Default | Description |
|---|---|---|---|
| `legend_loc` | `str` | `"upper right"` | Corner for legends: `"upper right"`, `"upper left"`, `"lower right"`, `"lower left"` |
| `figsize` | `tuple \| None` | `None` | Figure size (auto-calculated if `None`) |
| `color_present` | `str` | `"#2d2d2d"` | Colour for detected cells |
| `color_missing` | `str` | `"#f0f0f0"` | Colour for missing cells |

#### Font sizes

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fontsize` | `int` | `10` | Base font size (fallback) |
| `fontsize_legend` | `int \| None` | `None` | Legend entries |
| `fontsize_rows` | `int \| None` | `None` | Gene/row labels |
| `fontsize_cols` | `int \| None` | `None` | Sample/column labels |
| `fontsize_annotations` | `int \| None` | `None` | Annotation strip labels |

#### Group summary

| Parameter | Type | Default | Description |
|---|---|---|---|
| `group_summary` | `int \| str \| None` | `None` | Column level to group by; prints per-group completeness to console |

```python
fig = missing_matrix(df, group_summary="Medium_Condition")
```

Output:

```
Group Completeness (Medium_Condition)
--------------------------------
  SF               63%  (n=10)
  AS               80%  (n=10)
  FBS              88%  (n=10)
```

Only prints when the level has more than one group.

#### Split by factor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `split_by` | `int \| str \| None` | `None` | Split into side-by-side panels by this column level |

```python
fig = missing_matrix(df, split_by="Medium_Condition", annotation_levels=[0])
```

![split](output/demo_split.png)

Each panel is independently clustered. The split level is automatically removed from annotation strips.

#### Output

| Parameter | Type | Default | Description |
|---|---|---|---|
| `save` | `str \| None` | `None` | Save figure to this path |
| `dpi` | `int` | `150` | Save resolution |

## `missing_matrix_html()` -- interactive HTML

Plotly-based interactive version with hover tooltips showing feature name, sample ID, all annotation levels, and detection status.

```python
from mismap_qc import missing_matrix_html

missing_matrix_html(
    df,
    title="Gene Detection Matrix (Interactive)",
    subtitle="80 genes x 30 samples",
    completeness_threshold=0.5,
    save="output/interactive.html",
)
```

Supports the same clustering, sorting, annotation, and completeness options as the static version. Additional parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `width` | `int \| None` | `None` | Plot width in pixels (auto-calculated if `None`) |
| `height` | `int \| None` | `None` | Plot height in pixels (auto-calculated if `None`) |

Requires `plotly` (`pip install plotly` or included via PEP 723 in demo.py).

## Generating toy data

```bash
uv run make_toy_data.py
```

Creates `data/toy_rnaseq.csv`: 80 genes x 30 samples with structured missingness patterns across 6 groups (Fresh/Conditioned x SF/FBS/AS).

## Dependencies

- numpy
- matplotlib
- scipy
- pandas
- plotly (optional, for HTML export only)

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development install, how to run the tests and linter, and the conventions for adding a check or a plot. Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).

Development install:

```bash
git clone https://github.com/foertsch/mismap-qc.git
cd mismap-qc
uv sync --extra dev      # or: pip install -e ".[dev]"
```

## Citation

If you use mismap-qc in published work, please cite it. Machine-readable metadata is in [CITATION.cff](CITATION.cff); GitHub renders it as "Cite this repository" in the sidebar.

> Förtsch, A. (2026). *mismap-qc: missing-data validation for proteomics and RNA-Seq* (version 0.2.2). https://github.com/foertsch/mismap-qc

## License

MIT. See [LICENSE](LICENSE).
