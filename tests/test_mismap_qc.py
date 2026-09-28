"""Tests for mismap_qc.py"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # non-interactive backend for CI

sys.path.insert(0, str(Path(__file__).parent.parent))
from mismap_qc import (
    comissing_heatmap,
    completeness_bars,
    detection_waterfall,
    missing_matrix,
    missing_matrix_html,
    missing_mechanism,
    missing_runorder,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_flat_df(n_genes: int = 20, n_samples: int = 10, missing_frac: float = 0.2) -> pd.DataFrame:
    """Simple DataFrame with flat (non-MultiIndex) columns."""
    rng = np.random.default_rng(42)
    data = rng.random((n_genes, n_samples)).astype(float)
    mask = rng.random((n_genes, n_samples)) < missing_frac
    data[mask] = np.nan
    genes = [f"GENE_{i}" for i in range(n_genes)]
    samples = [f"S{i}" for i in range(n_samples)]
    return pd.DataFrame(data, index=genes, columns=samples)


def make_multiindex_df(
    n_genes: int = 20,
    n_per_group: int = 5,
    missing_frac: float = 0.2,
) -> pd.DataFrame:
    """DataFrame with a 2-level MultiIndex (Condition, Replicate)."""
    rng = np.random.default_rng(0)
    conditions = ["Fresh", "Conditioned"]
    tuples = [(cond, f"rep{r}") for cond in conditions for r in range(n_per_group)]
    columns = pd.MultiIndex.from_tuples(tuples, names=["Condition", "Replicate"])
    n_samples = len(tuples)
    data = rng.random((n_genes, n_samples)).astype(float)
    mask = rng.random((n_genes, n_samples)) < missing_frac
    data[mask] = np.nan
    return pd.DataFrame(data, index=[f"GENE_{i}" for i in range(n_genes)], columns=columns)


# ── missing_matrix (static) ───────────────────────────────────────────────────


def test_returns_figure_flat_df():
    import matplotlib.pyplot as plt

    fig = missing_matrix(make_flat_df())
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_returns_figure_multiindex_df():
    import matplotlib.pyplot as plt

    fig = missing_matrix(make_multiindex_df())
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_no_dendrogram():
    import matplotlib.pyplot as plt

    fig = missing_matrix(make_flat_df(), show_dendrogram=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_feature_type_options():
    """All feature_type options (PROT, GENE, PEPTIDE) should work."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    for ft in ["PROT", "GENE", "PEPTIDE"]:
        fig = missing_matrix(df, feature_type=ft)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


def test_no_clustering():
    import matplotlib.pyplot as plt

    fig = missing_matrix(make_flat_df(), cluster_samples=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_split_by():
    import matplotlib.pyplot as plt

    df = make_multiindex_df()
    fig = missing_matrix(df, split_by="Condition")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_save_to_disk(tmp_path: Path):
    import matplotlib.pyplot as plt

    out = tmp_path / "out.png"
    missing_matrix(make_flat_df(), save=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close("all")


def test_all_missing_column():
    """A column with all NaN values should not crash."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    df.iloc[:, 0] = np.nan
    fig = missing_matrix(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_all_present_column():
    """A column with no NaN values should not crash."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    df.iloc[:, 0] = 1.0
    fig = missing_matrix(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def _panel_heights(fig):
    """Gridspec height ratios of missing_matrix's stacked panels, top to bottom."""
    return list(fig.axes[0].get_subplotspec().get_gridspec().get_height_ratios())


def test_missing_matrix_dendrogram_is_bare():
    """No distance label, ticks or left spine: they collided with the strip below."""
    import matplotlib.pyplot as plt

    fig = missing_matrix(make_multiindex_df())
    dend = fig.axes[0]
    assert dend.get_ylabel() == ""
    assert len(dend.get_yticks()) == 0
    assert not dend.spines["left"].get_visible()
    plt.close("all")


def test_missing_matrix_panel_heights_small_data_unchanged():
    """Small matrices keep the old strip (0.4) and sparkline (1.2) heights."""
    import matplotlib.pyplot as plt

    fig = missing_matrix(make_multiindex_df(n_genes=20))
    # dendrogram, one annotation strip, matrix (floor of 6), sparkline
    assert _panel_heights(fig) == pytest.approx([2.0, 0.4, 6.0, 1.2])
    plt.close("all")


def test_missing_matrix_panel_heights_scale_with_tall_matrix():
    """On a tall matrix the strip and sparkline grow with it instead of
    shrinking to a few pixels."""
    import matplotlib.pyplot as plt

    fig = missing_matrix(make_multiindex_df(n_genes=1000))
    matrix_h = 1000 * 0.08
    assert _panel_heights(fig) == pytest.approx(
        [2.0, matrix_h * 0.02, matrix_h, matrix_h * 0.045])
    plt.close("all")


def test_missing_matrix_sample_labels_on_sparkline_when_below():
    """With the sparkline below, sample labels go on it, the bottom panel, so
    they cannot overlap it; the matrix carries none."""
    import matplotlib.pyplot as plt

    fig = missing_matrix(make_flat_df(), show_dendrogram=False)
    fig.canvas.draw()
    ax_mat, ax_sp = fig.axes[0], fig.axes[1]
    assert all(t.get_text() == "" for t in ax_mat.get_xticklabels())
    spark_labels = [t.get_text() for t in ax_sp.get_xticklabels()]
    assert sorted(spark_labels) == sorted(f"S{i}" for i in range(10))
    assert ax_sp.get_xlabel() == ""
    assert list(ax_sp.get_yticks()) == [0, 1]
    plt.close("all")


def test_missing_matrix_sample_labels_on_matrix_without_sparkline_below():
    """With the sparkline at the side, the matrix is the bottom panel and keeps
    the labels."""
    import matplotlib.pyplot as plt

    fig = missing_matrix(make_flat_df(), show_dendrogram=False, completeness="side")
    fig.canvas.draw()
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert sorted(labels) == sorted(f"S{i}" for i in range(10))
    plt.close("all")


def test_missing_matrix_many_samples_keep_samples_title():
    """Past 80 samples there are no tick labels, so the "Samples" title stays."""
    import matplotlib.pyplot as plt

    fig = missing_matrix(make_flat_df(n_samples=90), show_dendrogram=False,
                         cluster_samples=False)
    ax_sp = fig.axes[1]
    assert ax_sp.get_xlabel() == "Samples"
    fig.canvas.draw()
    assert all(t.get_text() == "" for t in ax_sp.get_xticklabels())
    plt.close("all")


# ── missing_matrix_html (interactive) ────────────────────────────────────────


def test_html_returns_string():
    pytest.importorskip("plotly")
    result = missing_matrix_html(make_flat_df())
    assert isinstance(result, str)
    assert "<div" in result


def test_html_save_to_disk(tmp_path: Path):
    pytest.importorskip("plotly")
    out = tmp_path / "interactive.html"
    missing_matrix_html(make_flat_df(), save=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


# ── missing_matrix invert ─────────────────────────────────────────────────────


def test_invert_swaps_colors():
    import matplotlib.pyplot as plt

    fig_normal = missing_matrix(make_flat_df())
    fig_inverted = missing_matrix(make_flat_df(), invert=True)
    assert isinstance(fig_normal, plt.Figure)
    assert isinstance(fig_inverted, plt.Figure)
    plt.close("all")


# ── completeness_bars ─────────────────────────────────────────────────────────


def test_completeness_bars_multiindex():
    import matplotlib.pyplot as plt

    fig = completeness_bars(make_multiindex_df(), group_level="Condition")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_flat_df():
    """Flat df → treated as single 'All samples' group."""
    import matplotlib.pyplot as plt

    fig = completeness_bars(make_flat_df(), group_level=0)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_threshold():
    import matplotlib.pyplot as plt

    fig = completeness_bars(make_multiindex_df(), group_level="Condition",
                            threshold=0.8)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_vertical():
    import matplotlib.pyplot as plt

    fig = completeness_bars(make_multiindex_df(), group_level="Condition",
                            orientation="vertical")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_custom_colors():
    import matplotlib.pyplot as plt

    colors = {"Fresh": "#88CCEE", "Conditioned": "#CC6677"}
    fig = completeness_bars(make_multiindex_df(), group_level="Condition",
                            color=colors)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_all_missing():
    """All-missing df should not crash (completeness = 0 for all groups)."""
    import matplotlib.pyplot as plt

    df = make_multiindex_df()
    df.iloc[:] = float("nan")
    fig = completeness_bars(df, group_level="Condition")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_all_present():
    """All-present df should not crash (completeness = 1 for all groups)."""
    import matplotlib.pyplot as plt

    df = make_multiindex_df()
    df = df.fillna(1.0)
    fig = completeness_bars(df, group_level="Condition")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_single_group():
    """Single-group MultiIndex should produce one bar without crashing."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(7)
    data = rng.random((20, 5))
    columns = pd.MultiIndex.from_tuples(
        [("OnlyGroup", f"s{i}") for i in range(5)],
        names=["Condition", "Sample"],
    )
    df = pd.DataFrame(data, columns=columns)
    fig = completeness_bars(df, group_level="Condition")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_save_to_disk(tmp_path: Path):
    import matplotlib.pyplot as plt

    out = tmp_path / "completeness.png"
    completeness_bars(make_multiindex_df(), group_level="Condition",
                      save=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close("all")


# ── detection_waterfall tests ─────────────────────────────────────────────────


def test_detection_waterfall_returns_figure():
    """Basic call returns a matplotlib Figure."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    fig = detection_waterfall(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_detection_waterfall_multiindex():
    """Works with MultiIndex columns."""
    import matplotlib.pyplot as plt

    df = make_multiindex_df()
    fig = detection_waterfall(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_detection_waterfall_with_groups():
    """Grouping by MultiIndex level produces multiple curves."""
    import matplotlib.pyplot as plt

    df = make_multiindex_df()
    fig = detection_waterfall(df, group_level="Condition")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_detection_waterfall_custom_thresholds():
    """Custom threshold values are applied."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    fig = detection_waterfall(df, thresholds=[0.3, 0.6])
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_detection_waterfall_no_thresholds():
    """Empty threshold list should not crash."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    fig = detection_waterfall(df, thresholds=[])
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_detection_waterfall_all_missing():
    """All-missing df should not crash."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    df.iloc[:] = float("nan")
    fig = detection_waterfall(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_detection_waterfall_all_present():
    """All-present df should show flat line at 100%."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    df = df.fillna(1.0)
    fig = detection_waterfall(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_detection_waterfall_save_to_disk(tmp_path: Path):
    """Save parameter writes PNG file."""
    import matplotlib.pyplot as plt

    out = tmp_path / "waterfall.png"
    detection_waterfall(make_flat_df(), save=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close("all")


def _ladder_df() -> pd.DataFrame:
    """Ten features over ten samples, feature i detected in 10 - i of them, so
    detection rates run 1.0, 0.9, ..., 0.1 and the threshold counts are known."""
    data = np.full((10, 10), np.nan)
    for i in range(10):
        data[i, : 10 - i] = 1.0
    return pd.DataFrame(data, index=[f"P{i}" for i in range(10)],
                        columns=[f"S{j}" for j in range(10)])


def _droplines(ax):
    """Vertical segments ending on the x-axis, as (x, top) pairs."""
    out = []
    for line in ax.get_lines():
        xs, ys = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        if len(xs) == 2 and xs[0] == xs[1] and ys[0] == 0:
            out.append((float(xs[0]), float(ys[1])))
    return sorted(out)


def test_detection_waterfall_droplines_at_threshold_counts():
    """Each threshold drops a line to the number of features at or above it."""
    import matplotlib.pyplot as plt

    fig = detection_waterfall(_ladder_df(), thresholds=[0.5, 0.9])
    # rates >= 0.9: 1.0 and 0.9 -> 2. Rates >= 0.5: 1.0 down to 0.5 -> 6.
    assert _droplines(fig.axes[0]) == [(2.0, 0.9), (6.0, 0.5)]
    plt.close("all")


def test_detection_waterfall_threshold_labels():
    """Labels give the count and say it is a detection cutoff, without a
    percentage of the total."""
    import matplotlib.pyplot as plt

    fig = detection_waterfall(_ladder_df(), thresholds=[0.5, 0.9], feature_type="GENE")
    texts = sorted(t.get_text() for t in fig.axes[0].texts)
    assert texts == ["2 genes at ≥90% detection", "6 genes at ≥50% detection"]
    plt.close("all")


def test_detection_waterfall_no_droplines_when_grouped():
    """With one curve per group, the pooled count is not where any curve
    crosses, so no dropline is drawn."""
    import matplotlib.pyplot as plt

    fig = detection_waterfall(make_multiindex_df(), group_level="Condition")
    assert _droplines(fig.axes[0]) == []
    plt.close("all")


@pytest.mark.parametrize("plot_fn", [detection_waterfall, missing_runorder])
@pytest.mark.parametrize("figsize", [(4, 3), (8, 5), (16, 10)])
def test_subtitle_sits_between_title_and_axes(plot_fn, figsize):
    """The subtitle is drawn fully below the title and above the axes, at any
    figure size. It used to print on top of the title."""
    import matplotlib.pyplot as plt

    fig = plot_fn(make_flat_df(), title="A title", subtitle="a subtitle",
                  figsize=figsize)
    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    fig.canvas.draw()
    title_box = ax.title.get_window_extent(renderer)
    (sub,) = [t for t in ax.texts if t.get_text() == "a subtitle"]
    sub_box = sub.get_window_extent(renderer)
    axes_top = ax.get_window_extent(renderer).y1
    assert sub_box.y1 <= title_box.y0
    assert sub_box.y0 >= axes_top
    plt.close("all")


def test_title_padding_unchanged_without_subtitle():
    """Without a subtitle the title keeps its old 10 pt pad."""
    import matplotlib.pyplot as plt

    for plot_fn in (detection_waterfall, missing_runorder):
        fig = plot_fn(make_flat_df(), title="A title")
        fig.canvas.draw()
        pad_pts = fig.axes[0].titleOffsetTrans.get_matrix()[1, 2] * 72 / fig.dpi
        assert pad_pts == pytest.approx(10)
        plt.close("all")


# ── missing_runorder tests ────────────────────────────────────────────────────


def test_missing_runorder_returns_figure():
    """Basic call returns a matplotlib Figure."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    fig = missing_runorder(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_runorder_multiindex():
    """Works with MultiIndex columns."""
    import matplotlib.pyplot as plt

    df = make_multiindex_df()
    fig = missing_runorder(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_runorder_with_groups():
    """Grouping by MultiIndex level colours points by group."""
    import matplotlib.pyplot as plt

    df = make_multiindex_df()
    fig = missing_runorder(df, group_level="Condition")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_runorder_custom_run_order():
    """Custom run_order array is used for x-axis."""
    import matplotlib.pyplot as plt

    df = make_flat_df(n_samples=10)
    run_order = list(range(100, 110))  # Custom x values
    fig = missing_runorder(df, run_order=run_order)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_runorder_no_smooth():
    """smooth=False disables the rolling mean line."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    fig = missing_runorder(df, smooth=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_runorder_small_sample():
    """Works with fewer samples than smooth window."""
    import matplotlib.pyplot as plt

    df = make_flat_df(n_samples=3)
    fig = missing_runorder(df, smooth=True, smooth_window=5)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_runorder_all_missing():
    """All-missing df should not crash."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    df.iloc[:] = float("nan")
    fig = missing_runorder(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_runorder_all_present():
    """All-present df should show points at 0%."""
    import matplotlib.pyplot as plt

    df = make_flat_df()
    df = df.fillna(1.0)
    fig = missing_runorder(df)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_runorder_save_to_disk(tmp_path: Path):
    """Save parameter writes PNG file."""
    import matplotlib.pyplot as plt

    out = tmp_path / "runorder.png"
    missing_runorder(make_flat_df(), save=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close("all")


def _two_batch_df(n_per_batch: int = 6) -> pd.DataFrame:
    """Ten features. Every B1 sample misses one of them (rate 0.1), every B2
    sample misses five (rate 0.5), so any smoothing across the boundary shows."""
    n = 2 * n_per_batch
    data = np.ones((10, n))
    data[:1, :n_per_batch] = np.nan
    data[:5, n_per_batch:] = np.nan
    cols = pd.MultiIndex.from_tuples(
        [("B1" if j < n_per_batch else "B2", f"r{j}") for j in range(n)],
        names=["Batch", "Run"])
    return pd.DataFrame(data, index=[f"P{i}" for i in range(10)], columns=cols)


def _smoother_lines(ax):
    """The rolling-mean lines, as (x, y) arrays."""
    from matplotlib.colors import to_hex

    return [(np.asarray(ln.get_xdata(), dtype=float), np.asarray(ln.get_ydata(), dtype=float))
            for ln in ax.get_lines() if to_hex(ln.get_color()) == "#cc4444"]


def test_missing_runorder_smoother_stays_within_groups():
    """Grouped, each batch gets its own smoother, so neither is pulled toward the
    other batch at the boundary. A single global smoother read 0.26 at the last
    B1 run."""
    import matplotlib.pyplot as plt

    fig = missing_runorder(_two_batch_df(), group_level="Batch")
    lines = _smoother_lines(fig.axes[0])
    assert len(lines) == 2
    (x1, y1), (x2, y2) = lines
    assert list(x1) == [0, 1, 2, 3, 4, 5] and np.allclose(y1, 0.1)
    assert list(x2) == [6, 7, 8, 9, 10, 11] and np.allclose(y2, 0.5)
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert legend.count("Rolling mean (n=5)") == 1
    plt.close("all")


def test_missing_runorder_ungrouped_smoother_is_one_line():
    """Without group_level there is still one smoother across all samples."""
    import matplotlib.pyplot as plt

    df = _two_batch_df().droplevel("Batch", axis=1)
    fig = missing_runorder(df)
    lines = _smoother_lines(fig.axes[0])
    assert len(lines) == 1
    x, y = lines[0]
    assert not np.isnan(x).any()
    assert y[5] == pytest.approx((0.1 * 3 + 0.5 * 2) / 5)
    plt.close("all")


def test_missing_runorder_smoother_breaks_across_run_order_gap():
    """A gap far wider than the usual spacing (here 95 against 1) lifts the pen:
    one NaN between the two runs of samples, both real endpoints kept."""
    import matplotlib.pyplot as plt

    df = _two_batch_df().droplevel("Batch", axis=1)
    run_order = [1, 2, 3, 4, 5, 6, 101, 102, 103, 104, 105, 106]
    fig = missing_runorder(df, run_order=run_order)
    ((x, y),) = _smoother_lines(fig.axes[0])
    nan_at = np.flatnonzero(np.isnan(x))
    assert list(nan_at) == [6]
    assert x[5] == 6 and x[7] == 101
    assert np.isnan(y[6])
    plt.close("all")


def test_missing_runorder_smoother_breaks_within_a_group():
    """The gap break applies inside a group too."""
    import matplotlib.pyplot as plt

    run_order = [1, 2, 3, 50, 51, 52, 200, 201, 202, 203, 204, 205]
    fig = missing_runorder(_two_batch_df(), group_level="Batch", run_order=run_order)
    (x1, _), (x2, _) = _smoother_lines(fig.axes[0])
    assert list(np.flatnonzero(np.isnan(x1))) == [3]
    assert not np.isnan(x2).any()
    plt.close("all")


def test_missing_runorder_contiguous_run_order_has_no_breaks():
    """Evenly spaced runs never trigger the gap break."""
    import matplotlib.pyplot as plt

    fig = missing_runorder(make_flat_df(n_samples=12))
    ((x, _),) = _smoother_lines(fig.axes[0])
    assert not np.isnan(x).any()
    plt.close("all")


def test_missing_runorder_skips_smoother_for_single_sample_group():
    """A group of one has nothing to smooth and gets no line."""
    import matplotlib.pyplot as plt

    df = _two_batch_df()
    cols = [("B3" if j == 11 else b, r) for j, (b, r) in enumerate(df.columns)]
    df.columns = pd.MultiIndex.from_tuples(cols, names=["Batch", "Run"])
    fig = missing_runorder(df, group_level="Batch")
    assert len(_smoother_lines(fig.axes[0])) == 2
    plt.close("all")


# ── missing_mechanism ─────────────────────────────────────────────────────────


def test_missing_mechanism_returns_figure_and_df():
    import matplotlib.pyplot as plt

    fig, classification = missing_mechanism(make_flat_df())
    assert isinstance(fig, plt.Figure)
    assert isinstance(classification, pd.DataFrame)
    assert set(classification.columns) >= {
        "feature", "mechanism", "missing_rate", "mean_abundance", "p_value",
    }
    assert set(classification["mechanism"].unique()) <= {
        "MNAR", "MAR", "INSUFFICIENT",
    }
    plt.close("all")


def test_missing_mechanism_multiindex():
    import matplotlib.pyplot as plt

    fig, classification = missing_mechanism(make_multiindex_df())
    assert isinstance(fig, plt.Figure)
    assert len(classification) > 0
    plt.close("all")


def test_missing_mechanism_no_scatter():
    import matplotlib.pyplot as plt

    fig, _ = missing_mechanism(make_flat_df(), show_scatter=False)
    assert isinstance(fig, plt.Figure)
    # Only one axes when scatter is off
    assert len(fig.axes) == 1
    plt.close("all")


def test_missing_mechanism_all_missing():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        np.full((10, 5), np.nan),
        index=[f"F{i}" for i in range(10)],
        columns=[f"S{i}" for i in range(5)],
    )
    fig, classification = missing_mechanism(df)
    assert isinstance(fig, plt.Figure)
    # All features have no observations -> all INSUFFICIENT
    assert (classification["mechanism"] == "INSUFFICIENT").all()
    plt.close("all")


def test_missing_mechanism_all_present():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        np.ones((10, 5)),
        index=[f"F{i}" for i in range(10)],
        columns=[f"S{i}" for i in range(5)],
    )
    fig, classification = missing_mechanism(df)
    assert isinstance(fig, plt.Figure)
    assert (classification["mechanism"] == "INSUFFICIENT").all()
    plt.close("all")


def test_missing_mechanism_invalid_method():
    with pytest.raises(ValueError, match="method must be"):
        missing_mechanism(make_flat_df(), method="quantile")


def test_missing_mechanism_save_to_disk(tmp_path):
    import matplotlib.pyplot as plt

    out = tmp_path / "mech.png"
    missing_mechanism(make_flat_df(), save=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close("all")


# ── comissing_heatmap ─────────────────────────────────────────────────────────


def test_comissing_heatmap_returns_figure():
    import matplotlib.pyplot as plt

    fig = comissing_heatmap(make_flat_df())
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_comissing_heatmap_multiindex():
    import matplotlib.pyplot as plt

    fig = comissing_heatmap(make_multiindex_df())
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_comissing_heatmap_no_cluster():
    import matplotlib.pyplot as plt

    fig = comissing_heatmap(make_flat_df(), cluster=False)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_comissing_heatmap_top_n_smaller_than_features():
    import matplotlib.pyplot as plt

    fig = comissing_heatmap(make_flat_df(n_genes=30), top_n=10)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_comissing_heatmap_all_missing():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        np.full((10, 5), np.nan),
        index=[f"F{i}" for i in range(10)],
        columns=[f"S{i}" for i in range(5)],
    )
    fig = comissing_heatmap(df, top_n=5)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_comissing_heatmap_all_present():
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        np.ones((10, 5)),
        index=[f"F{i}" for i in range(10)],
        columns=[f"S{i}" for i in range(5)],
    )
    fig = comissing_heatmap(df, top_n=5)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_comissing_heatmap_save_to_disk(tmp_path):
    import matplotlib.pyplot as plt

    out = tmp_path / "comiss.png"
    comissing_heatmap(make_flat_df(), save=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close("all")


def test_missing_mechanism_plots_only_categories_the_classifier_returns():
    """The bar chart must not show a category the classifier cannot produce.

    It used to draw an MCAR bar, permanently empty because the Mann-Whitney
    classifier never separates MCAR from MAR. An always-zero bar reads as
    "tested for MCAR and found none".
    """
    import matplotlib.pyplot as plt

    from mismap_qc import missing_mechanism

    fig, _ = missing_mechanism(make_flat_df(), show_scatter=False)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["MNAR", "MAR", "INSUFFICIENT"]
    plt.close("all")
