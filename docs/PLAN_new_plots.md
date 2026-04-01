# Plan: New Plot Functions for mismap-qc

Date: 2026-03-31
Updated: 2026-04-01 (detection_waterfall, missing_runorder done)
Status: In progress

---

## Overview

Nine new diagnostic plots to expand mismap-qc beyond the nullity matrix into a
complete missing-data QC toolkit for proteomics and transcriptomics. Priority order
is based on: (1) gap in existing Python tools, (2) direct actionability for users,
(3) implementation complexity.

**Wave 1** (current): `completeness_bars`, `detection_waterfall`, `missing_runorder`,
`missing_mechanism`, `comissing_heatmap`

**Wave 2**: `missing_upset`, `sample_outlier_score`, `batch_missing_test`,
`missing_summary_report`

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

---

## Plot 6 — UpSet Plot of Co-Missingness `missing_upset()`

**Priority: HIGH (Wave 2)**

### What it shows
UpSet plot of which combinations of samples share missing features. For each
intersection of samples (or groups), shows how many features are missing in
exactly that combination and no others.

### Why it matters
In small-n experiments (e.g. n=3 per group), knowing "do these two replicates
always lose the same proteins together?" is critical for distinguishing technical
dropout from biology. Bar charts can't show intersection structure; Venn diagrams
break down beyond 3 sets. UpSet is the right visualisation.

### No equivalent in Python
`upsetplot` exists but nobody wraps it for missing data specifically. This would
be a first.

### API sketch
```python
missing_upset(
    df,                          # features x samples, NaN = missing
    by="sample",                 # "sample" (one set per sample) or group label level
    min_size=1,                  # minimum intersection size to show
    title="Co-Missingness UpSet Plot",
    save=None,
    dpi=150,
) -> plt.Figure
```

### Layout
- Standard UpSet layout: bar chart on top (intersection size), matrix below
  (which samples are in that intersection), bar chart on left (per-sample
  total missingness)
- When `by` is a MultiIndex level name, one set per group rather than per sample

### Implementation notes
- Use `upsetplot` as a dependency (optional extra: `pip install mismap-qc[upset]`)
- Build the binary membership matrix from `df.isna()` then pass to `upsetplot.from_memberships()`
- For group mode: a feature counts as "missing in group X" if it is missing
  in ≥50% of samples in that group

---

## Plot 7 — Sample Outlier Score `sample_outlier_score()`

**Priority: HIGH (Wave 2)**

### What it shows
Per-sample missingness rate compared to other samples in the same group,
expressed as a z-score. Flags statistical outliers — samples whose missingness
is unexpectedly high or low relative to group peers.

### Why it matters
Users currently do this manually in Excel: compute per-sample % missing, sort,
eyeball. This function makes the decision data-driven and reproducible, and
returns a table they can act on directly (e.g. drop flagged samples before
imputation).

### No equivalent in Python
Nothing in `missingno` or any proteomics QC package does this.

### API sketch
```python
sample_outlier_score(
    df,                          # features x samples, NaN = missing
    group_level=None,            # MultiIndex level to compute z-scores within groups
    threshold=2.5,               # flag samples with |z| > threshold
    title="Sample Missingness Outlier Score",
    save=None,
    dpi=150,
) -> tuple[plt.Figure, pd.DataFrame]  # figure + per-sample table
```

### Layout
- One point per sample, x = sample index (or run order), y = per-sample
  missingness rate
- Points beyond threshold coloured red and labelled
- Dashed horizontal lines at mean ± threshold×SD
- If `group_level` set: compute z-scores within each group separately,
  colour points by group

### Returns
DataFrame with columns: `sample`, `group`, `missing_rate`, `z_score`, `flagged`

### Implementation notes
- `missing_rate = df.isna().mean(axis=0)`
- Z-score within group: `(x - group_mean) / group_std`
- Edge case: group with <3 samples → skip z-score, warn
- Reuse `_assign_colors()` for group colours

---

## Plot 8 — Batch Missingness Test `batch_missing_test()`

**Priority: MEDIUM (Wave 2)**

### What it shows
Volcano-style plot: for each feature, tests whether its missingness is
significantly enriched in one condition/batch vs another using Fisher's exact
test. X axis = log2 odds ratio, Y axis = -log10(p-value).

### Why it matters
A statistically principled alternative to eyeballing the matrix. Answers the
question "which proteins are specifically absent in condition X?" without
requiring imputation first. Currently not possible in any Python tool.

### No equivalent in Python
`protti` (R) has something similar. Nothing in Python.

### API sketch
```python
batch_missing_test(
    df,                          # features x samples, NaN = missing
    group_level,                 # MultiIndex level defining the two groups to compare
    group_a=None,                # label for group A (first group if None)
    group_b=None,                # label for group B (second group if None)
    alpha=0.05,                  # significance threshold
    min_missing=2,               # minimum missing count to test (avoids 0-cell tables)
    title="Batch Missingness Test",
    save=None,
    dpi=150,
) -> tuple[plt.Figure, pd.DataFrame]
```

### Layout
- Volcano plot: log2 OR on x, -log10(p) on y
- Colour: significant + enriched in A = blue, significant + enriched in B = red,
  non-significant = grey
- Dashed vertical lines at log2 OR = ±1, dashed horizontal at -log10(alpha)
- Label top N most significant features

### Returns
DataFrame with columns: `feature`, `log2_OR`, `p_value`, `q_value` (BH-corrected),
`significant`, `enriched_in`

### Implementation notes
- `scipy.stats.fisher_exact` per feature (2×2 table: missing/present × group A/B)
- BH correction via `statsmodels.stats.multitest.multipletests` or manual
- Only test features with ≥`min_missing` missing values across both groups combined
- Raise ValueError if group_level has more than 2 distinct values (only pairwise)

---

## Plot 9 — Summary QC Report `missing_summary_report()`

**Priority: LOW (Wave 2, longer-term)**

### What it shows
Generates a self-contained HTML report combining all mismap-qc plots for a
dataset into one document. Think MultiQC but specific to missing-data QC.

### Why it matters
Core facilities need shareable, standalone QC documents. One function call
that produces a complete HTML report — with all plots, a summary table, and
basic statistics — is the kind of feature that spreads a tool by word of mouth.

### No equivalent in Python
MultiQC exists but is pipeline-specific and not designed for custom missing-data
analysis.

### API sketch
```python
missing_summary_report(
    df,                          # features x samples, NaN = missing
    group_level=None,            # primary grouping for all group-aware plots
    output="mismap_qc_report.html",
    title="Missing Data QC Report",
    dpi=100,
) -> str  # path to saved HTML
```

### Layout
Sections (each collapible):
1. Dataset summary (shape, overall % missing, per-group table)
2. Missingness matrix (static PNG embedded inline)
3. Detection waterfall
4. Per-group completeness bars
5. Sample outlier scores (flagged samples highlighted)
6. Batch missingness test (if group_level provided)
7. Co-missingness heatmap

### Implementation notes
- Generate each plot as a PNG, base64-encode, embed in HTML template
- No external dependencies for the HTML (self-contained)
- Use Jinja2 template or simple f-string template
- Requires `plotly` for interactive matrix section (optional)
- Implement last — depends on all other functions being stable

---

## Implementation Order

### Wave 1
1. `completeness_bars()` — ✅ done
2. `detection_waterfall()` — moderate complexity, fills biggest Python gap
3. `missing_runorder()` — simple, useful for proteomics community
4. `missing_mechanism()` — most complex, highest scientific value
5. `comissing_heatmap()` — most niche, implement last

### Wave 2
6. `missing_upset()` — high impact for small-n experiments, needs `upsetplot`
7. `sample_outlier_score()` — highly actionable, straightforward to implement
8. `batch_missing_test()` — statistically rigorous, fills real gap
9. `missing_summary_report()` — longer-term, depends on all others being stable

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

### Wave 1
- [x] `completeness_bars()` — done
- [x] `detection_waterfall()` — done
- [x] `missing_runorder()` — done
- [ ] `missing_mechanism()` — not started
- [ ] `comissing_heatmap()` — not started

### Wave 2
- [ ] `missing_upset()` — not started
- [ ] `sample_outlier_score()` — not started
- [ ] `batch_missing_test()` — not started
- [ ] `missing_summary_report()` — not started (depends on Wave 1 completion)
