# Claude Code Instructions for mismap-qc

## Package Overview

**mismap-qc** is a Python library for **missing-data validation** in proteomics and RNA-Seq. It runs a battery of QC checks (sample completeness, outlier detection, MNAR/MAR mechanism classification, batch-effect tests, run-order drift), returns a structured `MismapReport`, and pairs every check with a publication-ready plot.

- **Author:** Arion Foertsch (FGCZ)
- **Package:** `mismap_qc/` (split across submodules; see Architecture)
- **Tests:** `tests/`
- **License:** MIT

---

## Architecture

### Package layout
As of 0.2.0 the project is split from a single `mismap_qc.py` into a `mismap_qc/` package:

| Module | Lines | Contents |
|---|---|---|
| `mismap_qc/__init__.py` | ~100 | Public re-exports. `from mismap_qc import qc, missing_matrix, ...` works the same way. |
| `mismap_qc/_core.py` | ~60 | Constants and layout helpers: `_PALETTES`, `FEATURE_TYPES`, `_get_feature_labels`, `_assign_colors`, `_clean_ax`, `_resolve_color_overrides`. |
| `mismap_qc/stats.py` | ~190 | Pure-numeric analytical helpers shared by `qc()` and the plot functions: `_classify_mechanism`, `_top_codropouts`, `_comissing_matrix`, `_batch_missing_test`, `_runorder_trend`, `_compute_sample_outliers`, `_resolve_group_labels`. |
| `mismap_qc/validation.py` | ~615 | Validation API: `MismapReport`, `RuleResult`, `MismapQCWarning`, `MismapQCFailure`, the 11-rule registry, `_evaluate_thresholds`, `qc()`, `assert_qc()`. |
| `mismap_qc/plots.py` | ~1855 | All plot functions (`missing_matrix`, `missing_matrix_html`, `completeness_bars`, `detection_waterfall`, `missing_runorder`, `missing_mechanism`, `comissing_heatmap`, `missing_abundance_density`) plus the `_data_*` helpers and `_RETURN_DATA_SCHEMAS` registry that backs `return_data=True`. |
| `mismap_qc/io.py` | ~155 | `from_anndata()` reader. `anndata` is an optional dependency. |
| `mismap_qc/lod.py` | ~60 | `estimate_lod()`. Future Scope E items go here. |

### Dependency graph
```
_core  <-  plots
       <-  __init__
stats  <-  plots
       <-  validation
       <-  __init__
io, lod  <-  __init__
```
No circular imports. Plots and validation both depend on stats; neither depends on the other.

### Function categories
1. **Validation entry points** — `qc()`, `assert_qc()`, plus the `MismapReport` it returns.
2. **Primary plots** — User-facing visualization functions.
3. **Analytical helpers** — Underscore-prefixed numerical functions in `stats.py`; shared between `qc()` and plot wrappers (e.g. `missing_mechanism` wraps `_classify_mechanism`).
4. **Internal layout helpers** — Underscore-prefixed in `_core.py` (`_assign_colors`, `_clean_ax`, etc.).

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

### Wave 1 (complete in 0.2.0)
- [x] `missing_matrix()` — Main nullity matrix
- [x] `missing_matrix_html()` — Interactive HTML version
- [x] `completeness_bars()` — Per-group completeness
- [x] `detection_waterfall()` — Feature detection curve
- [x] `missing_runorder()` — Missingness over time
- [x] `missing_mechanism()` — MNAR/MAR classification (wraps `_classify_mechanism`)
- [x] `comissing_heatmap()` — Co-missingness patterns (wraps `_comissing_matrix`)

### Validation API (new in 0.2.0)
- [x] `qc()` / `assert_qc()` / `MismapReport`
- [x] 11 threshold rules with error / warning / info severities
- [x] `return_data=True` flag on every plot
- [x] `from_anndata()` interop reader
- [x] `estimate_lod()` (Scope E.1)

### Wave 2 (planned, post-resubmission)
- [ ] `missing_upset()` — UpSet plot of co-missingness
- [ ] `sample_outlier_score()` — Outlier detection (currently inline in `qc()`)
- [ ] `batch_missing_test()` — Statistical batch comparison (currently inline in `qc()`)
- [ ] `missing_summary_report()` — HTML report generator
- [ ] CLI (`mismap-qc qc data.tsv --thresholds rules.yml`)
- [ ] Scope E.2/E.3 (`imputation_diagnostic`, `replicate_concordance`)
- [ ] Search-engine parsers (`from_maxquant`, `from_diann`, `from_fragpipe`, `from_spectronaut`)

See `docs/PLAN_new_plots.md` and `docs/PLAN_validation_scope.md` for specs.

---

## Files to Know

| File | Purpose |
|------|---------|
| `mismap_qc/` | The package. See Architecture for per-module breakdown. |
| `tests/test_mismap_qc.py` | Plot function tests (synthetic data) |
| `tests/test_validation_api.py` | qc / assert_qc / MismapReport tests |
| `tests/test_return_data_schemas.py` | Schema regression test for `return_data=True` |
| `tests/test_from_anndata.py` | AnnData reader tests |
| `tests/test_estimate_lod.py` | LOD estimator tests |
| `tests/conftest.py` | Shared CPTAC fixture (real-data tests skipped if absent) |
| `examples/cptac_proteomics.ipynb` | Real-world tutorial (still uses v0.1 API; update pending) |
| `docs/PLAN_new_plots.md` | Wave 1/2 plot roadmap |
| `docs/PLAN_validation_scope.md` | Validation API + interop spec |
| `CHANGELOG.md` | Per-version change log |
| `demo.py` | Quick demo script (PEP 723) |

---

## Do's and Don'ts

### Do
- Add `feature_type` parameter to any function that labels features
- Use `_get_feature_labels()` for all user-facing text
- Put new code in the right submodule (plots, stats, validation, io, lod). When in doubt, plots.py is the default for figure-returning functions; stats.py for numeric-returning ones
- Re-export new public names in `mismap_qc/__init__.py`
- Add `return_data=True` support to any new plot function, with the schema registered in `_RETURN_DATA_SCHEMAS`
- Add the matching schema test in `tests/test_return_data_schemas.py`
- Include edge case tests (all-missing, all-present, MultiIndex)
- Follow the function signature template

### Don't
- Don't hardcode "gene" or "protein" in labels
- Don't duplicate analytical logic between `qc()` and plot functions — share via `stats.py`
- Don't add new rules to `_RULE_DEFAULT_SEVERITY` without also adding an evaluator in `_RULE_EVALUATORS` and a test
- Don't skip `plt.close("all")` in tests
- Don't use `sort_genes` (renamed to `sort_features`)
- Don't break backwards compatibility without deprecation warnings
- Don't reintroduce `mismap_qc.py` as a single file — the project crossed that threshold at 0.2.0
