# Claude Code Instructions for mismap-qc

## Package Overview

**mismap-qc** is a Python visualization library for missing-data QC in proteomics and RNA-Seq. It produces publication-ready plots showing which features (proteins/genes/peptides) are detected vs missing across samples.

- **Author:** Arion Foertsch (FGCZ)
- **Main module:** `mismap_qc.py` (~1,500 lines, single file)
- **Tests:** `tests/test_mismap_qc.py`
- **License:** MIT

---

## Architecture

### Single-file design
All functions live in `mismap_qc.py`. This is intentional for easy installation (`pip install mismap-qc` or just copy the file). Don't split into submodules unless it exceeds ~3,000 lines.

### Function categories
1. **Primary plots** — User-facing visualization functions
2. **Helper functions** — Prefixed with `_` (e.g., `_assign_colors`, `_get_feature_labels`)
3. **Internal** — `_split_matrix` for split-by-factor rendering

---

## Coding Patterns

### Function signature template
```python
def new_plot_function(
    df: pd.DataFrame,
    *,                              # Force keyword args after df
    param1: str = "default",
    feature_type: str = "PROT",     # Always include
    title: str | None = None,       # None = auto-generate from feature_type
    subtitle: str = "",
    figsize: tuple[float, float] | None = None,
    fontsize: int = 10,
    save: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
```

### Feature type handling
Always use the `FEATURE_TYPES` dict and `_get_feature_labels()`:
```python
fl = _get_feature_labels(feature_type)
# fl['singular']     -> "protein"
# fl['plural']       -> "proteins"
# fl['cap_singular'] -> "Protein"
# fl['cap_plural']   -> "Proteins"

# Use in labels:
ax.set_xlabel(f"{fl['cap_plural']} (ranked by detection)", fontsize=fontsize)
ax.text(..., f"{n} {fl['plural']} at ≥{thresh:.0%}")
```

### MultiIndex handling
```python
has_mi = isinstance(df.columns, pd.MultiIndex)

# Resolve string level names to int indices:
if isinstance(group_level, str):
    grp_lv = list(df.columns.names).index(group_level)
else:
    grp_lv = group_level
```

### Color assignment
Use `_assign_colors()` for consistent palettes across plots:
```python
_, cmap = _assign_colors(np.array(group_labels), palette_idx=0)
color = cmap[group_name]  # Returns hex string
```

### Figure creation pattern
```python
if figsize is None:
    figsize = (8, 5)  # Sensible default

fig, ax = plt.subplots(figsize=figsize, facecolor="white")

# ... plotting code ...

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

if title:
    ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=10)

fig.tight_layout()

if save:
    fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")

return fig
```

---

## Parameter Naming Conventions

| Pattern | Usage |
|---------|-------|
| `df` | Always the first parameter, features × samples DataFrame |
| `feature_type` | "PROT", "GENE", or "PEPTIDE" |
| `group_level` | int or str for MultiIndex level |
| `threshold` | Single threshold value (float 0-1) |
| `thresholds` | List of threshold values |
| `save` | Output file path (None = don't save) |
| `dpi` | Save resolution (default 150) |
| `fontsize` | Base font size (default 10) |
| `figsize` | Tuple or None for auto |

---

## Testing Conventions

### Test file structure
- `tests/test_mismap_qc.py` — Synthetic data tests (run in CI)
- `tests/test_*_realdata.py` — Real CPTAC data tests (skipped if data missing)
- `tests/conftest.py` — Shared fixtures

### Required tests for each function
1. `test_<func>_returns_figure` — Basic call returns plt.Figure
2. `test_<func>_multiindex` — Works with MultiIndex columns
3. `test_<func>_all_missing` — Handles all-NaN data without crash
4. `test_<func>_all_present` — Handles complete data without crash
5. `test_<func>_save_to_disk` — `save` parameter writes file

### Test pattern
```python
def test_new_function_returns_figure():
    import matplotlib.pyplot as plt

    df = make_flat_df()  # From fixtures
    fig = new_function(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")  # Always close figures
```

---

## Development Status

### Wave 1 (current)
- [x] `missing_matrix()` — Main nullity matrix
- [x] `missing_matrix_html()` — Interactive HTML version
- [x] `completeness_bars()` — Per-group completeness
- [x] `detection_waterfall()` — Feature detection curve
- [x] `missing_runorder()` — Missingness over time
- [ ] `missing_mechanism()` — MNAR/MAR classification
- [ ] `comissing_heatmap()` — Co-missingness patterns

### Wave 2 (planned)
- [ ] `missing_upset()` — UpSet plot of co-missingness
- [ ] `sample_outlier_score()` — Outlier detection
- [ ] `batch_missing_test()` — Statistical batch comparison
- [ ] `missing_summary_report()` — HTML report generator

See `docs/PLAN_new_plots.md` for full specifications.

---

## Files to Know

| File | Purpose |
|------|---------|
| `mismap_qc.py` | All plot functions |
| `tests/test_mismap_qc.py` | Main test suite |
| `tests/conftest.py` | Shared test fixtures (cptac_df) |
| `examples/cptac_proteomics.ipynb` | Real-world tutorial |
| `docs/PLAN_new_plots.md` | Implementation roadmap |
| `demo.py` | Quick demo script (PEP 723) |

---

## Do's and Don'ts

### Do
- Add `feature_type` parameter to any function that labels features
- Use `_get_feature_labels()` for all user-facing text
- Include edge case tests (all-missing, all-present)
- Follow the function signature template
- Keep functions in `mismap_qc.py` (single file)

### Don't
- Don't hardcode "gene" or "protein" in labels
- Don't create new submodules without discussion
- Don't skip `plt.close("all")` in tests
- Don't use `sort_genes` (renamed to `sort_features`)
- Don't break backwards compatibility without deprecation warnings
