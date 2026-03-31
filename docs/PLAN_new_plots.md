# Plan: New Plot Functions for mismap-qc

Date: 2026-03-31
Status: Planning

---

## Overview

Five new diagnostic plots to expand mismap-qc beyond the nullity matrix into a more
complete missing-data QC toolkit for proteomics and transcriptomics. Priority order
is based on: (1) gap in existing Python tools, (2) direct actionability for users,
(3) implementation complexity.

---

## Plot 1 — Detection Threshold Waterfall `detection_waterfall()`

**Priority: HIGH**

### What it shows
Genes/proteins ranked by % detection across all samples, plotted as a cumulative
curve. Horizontal threshold lines show how many features survive at different
filtering cutoffs (e.g. "detected in ≥50% of samples").

### Why it matters
Every RNA-Seq and proteomics analysis starts with a filtering step: "keep genes
detected in at least N samples". Nobody visualises the tradeoff. This plot makes
the filtering decision explicit and defensible.

### No equivalent in Python
`DEP` (R) has a bar chart version. Nothing in Python does this.

### API sketch
```python
detection_waterfall(
    df,                          # features x samples, NaN = missing
    thresholds=[0.5, 0.7, 0.9], # draw lines at these detection rates
    groups=None,                 # if set, compute per-group detection rates
    color="#2d2d2d",
    title="Feature Detection Waterfall",
    save=None,
    dpi=150,
) -> plt.Figure
```

### Layout
- X axis: features ranked by detection rate (descending)
- Y axis: % detected across samples (0–100%)
- Filled area under curve
- Dashed horizontal lines at each threshold value
- Annotations: "N features at ≥X% detection"
- If `groups` provided: one curve per group, same axes

### Implementation notes
- Simple to implement: sort `df.notna().mean(axis=1)`, plot as line/fill
- Group curves: compute `df.loc[:, group_mask].notna().mean(axis=1)` per group
- Use same palette as `_PALETTES` for group curves
- Edge case: all features 100% detected → still useful (flat line at top)

---

## Plot 2 — MNAR/MAR Classification Plot `missing_mechanism()`

**Priority: HIGH**

### What it shows
For each feature (protein/gene), classifies its missing data mechanism:
- **MNAR** (Missing Not At Random): missing values correlate with low abundance
- **MAR** (Missing At Random): missingness is independent of abundance
- **MCAR** (Missing Completely At Random): random, no pattern

Outputs a summary bar chart + optional scatter showing each feature's classification.

### Why it matters
This is the key decision point before imputation. MNAR proteins need downshift
imputation (Perseus-style). MAR proteins can use KNN or median. Getting this wrong
introduces systematic bias. `protti` does this in R; no Python equivalent exists.

### API sketch
```python
missing_mechanism(
    df,                          # features x samples, NaN = missing
    method="lm",                 # "lm" (linear model) or "quantile"
    alpha=0.05,                  # significance threshold for MNAR test
    min_present=3,               # minimum non-missing values to test
    title="Missing Data Mechanism",
    show_scatter=True,           # scatter of mean abundance vs missing rate
    save=None,
    dpi=150,
) -> tuple[plt.Figure, pd.DataFrame]  # figure + per-feature classification table
```

### Classification logic
For each feature:
1. Compute mean abundance of non-missing values
2. Compare mean abundance of samples where feature IS present vs all samples
3. If missing values cluster at low abundance end → MNAR (test: t-test or
   Mann-Whitney U on abundance of present vs "would-be" missing, proxied by
   overall abundance rank)
4. Practical implementation: for each feature, test whether the bottom Q quantile
   of samples accounts for disproportionate missingness (similar to protti's approach)

### Layout
Two-panel figure:
- **Left**: horizontal bar chart — count of MNAR / MAR / MCAR features
- **Right** (if `show_scatter=True`): scatter of mean abundance (x) vs missing rate (y),
  coloured by classification. Shows the MNAR "L-shape" pattern visually.

### Returns
Also returns a DataFrame with columns: `feature`, `mechanism`, `missing_rate`,
`mean_abundance`, `p_value` — so users can filter their data programmatically.

### Implementation notes
- Use scipy `mannwhitneyu` for the abundance comparison
- Bin features with <`min_present` non-missing values as "insufficient data"
- Colour scheme: MNAR = red, MAR = orange, MCAR = blue (standard in literature)
- This is the most complex function in this plan — implement last despite high priority

---

## Plot 3 — Per-Group Completeness Bar Chart `completeness_bars()`

**Priority: MEDIUM**

### What it shows
Horizontal bar chart of per-group detection completeness (% features detected),
one bar per group. Replaces the current `group_summary` console print with a
publishable figure.

### Why it matters
`group_summary` is currently console-only. A figure version can be included in
QC reports and publications. Very fast to implement.

### API sketch
```python
completeness_bars(
    df,                          # features x samples, NaN = missing
    group_level,                 # int or str — MultiIndex level to group by
    threshold=None,              # draw a dashed line at this completeness value
    color=None,                  # single colour or dict {group: colour}
    orientation="horizontal",    # "horizontal" or "vertical"
    title="Per-Group Completeness",
    save=None,
    dpi=150,
) -> plt.Figure
```

### Layout
- One bar per group, length = mean % features detected
- Bars sorted descending by completeness
- Threshold line (dashed red) if provided
- Bar labels showing exact % value
- Colour-coded by group (same colours as annotation strips in `missing_matrix`)

### Implementation notes
- Straightforward: group columns by `group_level`, compute `notna().mean().mean()`
- Should reuse `_assign_colors()` for consistent colours with the main matrix
- Also add `return_data=True` option to return the completeness values as a Series

---

## Plot 4 — Co-Missingness Heatmap `comissing_heatmap()`

**Priority: MEDIUM-LOW**

### What it shows
Heatmap of pairwise co-missingness between features: how often are two features
missing in the same sample? Reveals systematic co-dropout of protein complexes,
pathways, or instrument failure patterns.

### Why it matters
If two proteins always go missing together, they likely share a biological or
technical reason. Useful for identifying: (1) co-regulated low-abundance proteins,
(2) systematic instrument failures affecting specific m/z ranges, (3) peptide
co-elution issues.

### API sketch
```python
comissing_heatmap(
    df,                          # features x samples, NaN = missing
    top_n=50,                    # only show top N features by missingness rate
    cluster=True,                # cluster features by co-missingness pattern
    method="average",            # linkage method
    cmap="Blues",
    title="Co-Missingness Heatmap",
    save=None,
    dpi=150,
) -> plt.Figure
```

### Layout
- Square heatmap: features × features
- Cell value = fraction of samples where both features are missing simultaneously
- Hierarchical clustering on both axes if `cluster=True`
- Diagonal = per-feature missingness rate (special colour)
- Dendrogram on both axes

### Implementation notes
- Compute co-missingness matrix: `M = df.isna().astype(int)`, then `(M @ M.T) / n_samples`
- Limit to `top_n` features by missingness rate to keep it readable
- Use `seaborn.clustermap` or build with `ComplexHeatmap`-style gridspec
- Diagonal handling: set diagonal to NaN and use a different colour

---

## Plot 5 — Missingness Over Run Order `missing_runorder()`

**Priority: LOW**

### What it shows
% missing per sample plotted against sample run order or collection date.
Instrument drift shows up as a trend; batch effects show up as jumps.

### Why it matters
In large proteomics experiments (>50 samples), instrument performance degrades
over time. This is the first plot any proteomics core facility runs. No Python tool
does this specifically for missingness (only for intensity).

### API sketch
```python
missing_runorder(
    df,                          # features x samples, NaN = missing
    run_order=None,              # list/Series of run order values; uses column order if None
    groups=None,                 # colour points by group (e.g. batch, condition)
    smooth=True,                 # add LOESS/rolling mean smoother
    title="Missingness Over Run Order",
    save=None,
    dpi=150,
) -> plt.Figure
```

### Layout
- X axis: run order (or column index if not provided)
- Y axis: % missing per sample
- Points coloured by `groups` if provided
- Smoother line (rolling mean, window=5) if `smooth=True`
- Dashed horizontal line at dataset mean missingness

### Implementation notes
- Simple: `df.isna().mean(axis=0)` gives per-sample missingness
- Rolling mean via `pd.Series.rolling(window=5, center=True).mean()`
- If `groups` provided, reuse `_assign_colors()`
- Most useful for proteomics; RNA-Seq rarely has meaningful run order

---

## Implementation Order

Given complexity vs. impact:

1. `completeness_bars()` — easiest, high visibility, replaces console output
2. `detection_waterfall()` — moderate complexity, fills biggest Python gap
3. `missing_runorder()` — simple, useful for proteomics community
4. `missing_mechanism()` — most complex, highest scientific value
5. `comissing_heatmap()` — most niche, implement last

---

## Testing Requirements

### Synthetic tests (always run, including CI)

Each new function needs pytest tests covering:
- Returns a `plt.Figure`
- Works with flat columns and MultiIndex columns
- Handles edge cases: all-present, all-missing, single group
- `save` parameter writes a file to `tmp_path`
- Where applicable: returned DataFrame has expected columns

### Real-data tests (local only, skip in CI)

The CPTAC LUAD data in `examples/data/` is used to validate each function on
real proteomics data. These tests are skipped automatically if the data files
are not present (so CI stays green without committing large data files).

**Pattern to follow for every new function:**

`tests/conftest.py` — shared fixture:
```python
import pytest
import pandas as pd
from pathlib import Path

CPTAC_TUMOR = Path(__file__).parent.parent / "examples/data/HS_CPTAC_LUAD_proteome_ratio_NArm_TUMOR.cct"
CPTAC_NORMAL = Path(__file__).parent.parent / "examples/data/HS_CPTAC_LUAD_proteome_ratio_NArm_NORMAL.cct"

@pytest.fixture(scope="session")
def cptac_df():
    """Load CPTAC LUAD combined proteomics matrix. Skip if data not present."""
    if not CPTAC_TUMOR.exists() or not CPTAC_NORMAL.exists():
        return None
    tumor = pd.read_csv(CPTAC_TUMOR, sep="\t", index_col=0)
    normal = pd.read_csv(CPTAC_NORMAL, sep="\t", index_col=0)
    tumor.columns = [f"{c}_T" for c in tumor.columns]
    normal.columns = [f"{c}_N" for c in normal.columns]
    return pd.concat([tumor, normal], axis=1)

@pytest.fixture(scope="session")
def cptac_groups(cptac_df):
    if cptac_df is None:
        return None
    return ["Tumor" if c.endswith("_T") else "Normal" for c in cptac_df.columns]
```

**Usage in test files:**
```python
def test_completeness_bars_real_data(cptac_df):
    if cptac_df is None:
        pytest.skip("CPTAC data not available")
    # build MultiIndex and run function...
    fig = completeness_bars(df_with_multiindex, group_level="Tumor_Normal")
    assert isinstance(fig, plt.Figure)
    plt.close("all")
```

**Rules:**
- Real-data tests go in a separate file: `tests/test_*_realdata.py`
- Never commit the `.cct` or `.tsi` data files (already in `.gitignore` via `examples/data/`)
- Real-data tests run automatically when data is present; skipped otherwise
- CI only runs the synthetic tests

---

## Documentation / Tutorial Updates

After implementation:
- Add each function to the CPTAC notebook with a real proteomics example
- Update README with a new "Functions" section listing all available plots
- Update pyOpenSci submission to reflect expanded scope

---

## Status

- [ ] `completeness_bars()` — not started
- [ ] `detection_waterfall()` — not started
- [ ] `missing_runorder()` — not started
- [ ] `missing_mechanism()` — not started
- [ ] `comissing_heatmap()` — not started
