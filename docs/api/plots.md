# Plots

Every function here accepts `return_data=True` and then returns `(Figure, DataFrame)` instead of a bare figure. Schemas are registered in `_RETURN_DATA_SCHEMAS` and pinned by a regression test, so they are part of the public contract.

| Function | `return_data` columns |
|---|---|
| `missing_matrix` | `feature`, `sample`, `missing` |
| `completeness_bars` | `group`, `completeness`, `n_samples` |
| `detection_waterfall` | `feature`, `detection_rate`, `rank` |
| `missing_runorder` | `sample`, `run_order`, `missing_rate`, `group` |
| `comissing_heatmap` | `feature_a`, `feature_b`, `comissingness` |
| `missing_upset` | `feature`, `intersection_id`, `members`, `n_features`, `rank`, `plotted` |

## The nullity matrix

::: mismap_qc.plots.missing_matrix

::: mismap_qc.plots.missing_matrix_html

## Per-check plots

::: mismap_qc.plots.completeness_bars

::: mismap_qc.plots.detection_waterfall

::: mismap_qc.plots.missing_runorder

::: mismap_qc.plots.missing_mechanism

::: mismap_qc.plots.comissing_heatmap

::: mismap_qc.plots.missing_upset

::: mismap_qc.plots.missing_abundance_density
