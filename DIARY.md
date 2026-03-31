# pretty-missing development diary

## 2026-03-11: Initial build

### What is this?
A challenge from a friend to make a prettier version of the `missingno` matrix plot,
tailored for RNA-Seq data. Shows which genes are detected vs missing across samples,
with multi-level sample annotations and hierarchical clustering.

### Core function: `missing_matrix(df, ...)`

Input: a pandas DataFrame with genes as rows, samples as MultiIndex columns.
NaN = missing/not detected. Level names on the MultiIndex automatically become
annotation strip labels.

### Features implemented

**Layout (top to bottom):**
- Title (bold, configurable)
- Subtitle -- secondary italic line for dataset metadata (n samples, missingness %, etc.)
- Dendrogram -- clusters samples by binary nullity pattern (scipy hierarchical clustering)
- Annotation colour strips -- one per column level, auto-derived from MultiIndex
- Nullity matrix -- dark = detected, light = missing (imshow, vectorised RGB)
- Completeness sparkline -- per-sample or per-gene detection rate
- Completeness threshold line -- dashed red line at a user-defined cutoff

**Clustering & sorting:**
- `cluster_samples=True` -- hierarchical clustering on binary nullity (configurable method)
- `kmeans=k` -- alternative: k-means clustering with k clusters. Samples ordered by
  cluster assignment then by completeness within each cluster. Dendrogram auto-disabled.
- `sort_genes="descending"` -- sort genes by completeness (most complete at top)
- Dendrogram can be hidden with `show_dendrogram=False`

**Annotations:**
- `annotation_levels` -- choose which column levels to show (default: all except innermost)
- `annotation_colors` -- custom colour mapping per level, keyed by level index (int) or
  level name (str). Maps factor values to hex colours. Missing factors fall back to
  built-in palettes. Example:
  ```python
  annotation_colors={
      "Medium_Type": {"Fresh": "#88CCEE", "Conditioned": "#CC6677"},
      "Medium_Condition": {"SF": "#44AA99", "FBS": "#DDCC77", "AS": "#AA4499"},
  }
  ```

**Completeness sparkline:**
- `completeness="below"` -- horizontal sparkline under the matrix (per-sample completeness)
- `completeness="side"` -- vertical sparkline to the right (per-gene completeness)
- `completeness_threshold=0.5` -- draws a dashed red line at 50% on the sparkline

**Legends:**
- All legends (annotation levels + detected/missing) stacked tightly in one corner
- `legend_loc` -- `"upper right"`, `"upper left"`, `"lower right"`, `"lower left"`
- Legends are measured and positioned with minimal gap (no wasted space)

**Per-group completeness summary:**
- `group_summary=True` -- renders a monospace table in the opposite corner from the
  legends showing completeness % and sample count per annotation group
- Example output: `SF  63%  (n=10)`, `FBS  88%  (n=10)`

**Split by factor:**
- `split_by="Medium_Condition"` -- renders one panel per factor level, arranged
  side by side. Each panel is independently clustered. The split level is auto-removed
  from annotation strips (no redundant info).

**Font size controls:**
- `fontsize` -- base font size (fallback for everything)
- `fontsize_legend` -- legend entries
- `fontsize_rows` -- gene/row labels
- `fontsize_cols` -- sample/column labels
- `fontsize_annotations` -- annotation strip labels

**Other:**
- `color_present` / `color_missing` -- customise matrix colours
- `figsize` -- manual or auto-calculated
- `save` / `dpi` -- save to file
- `label_level` -- which column level to use for x-axis tick labels (default: innermost)
- Works with plain (non-MultiIndex) DataFrames too (just no annotation strips)

### Interactive HTML: `missing_matrix_html(df, ...)`

- Plotly-based interactive version with hover tooltips
- Shows gene name, sample ID, all annotation levels, and detection status on hover
- Annotation strips rendered as coloured heatmap rows
- Completeness bar chart below the matrix
- Supports same clustering options (hierarchical, k-means)
- Exports self-contained HTML (plotly CDN)
- `save="output.html"` writes to file

### Repo structure
```
pretty-missing/
  pretty_missing.py    -- the functions (PEP 723 inline deps, no venv needed)
  make_toy_data.py     -- generates synthetic RNA-Seq data with structured missingness
  demo.py              -- loads toy data, demos all features, saves 4 outputs
  data/toy_rnaseq.csv  -- pre-generated toy dataset (80 genes x 30 samples)
  output/              -- generated plots (gitignored)
  DIARY.md             -- this file
```

### TODO / ideas for later
- Row-side annotation strips (e.g. gene ontology categories, pathway membership)
- Support for passing an external dendrogram / linkage matrix
- Colour dendrogram branches by cluster threshold
- Configurable sparkline colour (separate from matrix colours)
- Transpose mode (samples as rows, genes as columns)
- Integration with scanpy/AnnData (accept AnnData, pull annotations from .obs)
- Export to SVG with publication-friendly defaults
