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
    completeness_violin,
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


# ── completeness_violin ───────────────────────────────────────────────────────


def _violin_df() -> pd.DataFrame:
    """Ten features. Group A samples are 100%, 80% and 60% complete (mean 80%);
    both group B samples are 50% complete, so B has no spread to draw a KDE of."""
    data = np.ones((10, 5))
    data[:2, 1] = np.nan   # A2: 8 of 10
    data[:4, 2] = np.nan   # A3: 6 of 10
    data[:5, 3] = np.nan   # B1: 5 of 10
    data[5:, 4] = np.nan   # B2: 5 of 10, the other half
    cols = pd.MultiIndex.from_tuples(
        [("A", "A1"), ("A", "A2"), ("A", "A3"), ("B", "B1"), ("B", "B2")],
        names=["Condition", "Sample"])
    return pd.DataFrame(data, index=[f"P{i}" for i in range(10)], columns=cols)


def _violin_bodies(ax):
    from matplotlib.collections import PathCollection, PolyCollection

    return [c for c in ax.collections
            if isinstance(c, PolyCollection) and not isinstance(c, PathCollection)]


def _scatter_points(ax):
    from matplotlib.collections import PathCollection

    return [c for c in ax.collections if isinstance(c, PathCollection)]


def test_completeness_violin_returns_figure():
    import matplotlib.pyplot as plt

    fig = completeness_violin(make_flat_df(), group_level=0)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_violin_multiindex():
    """One violin position per group of the chosen level."""
    import matplotlib.pyplot as plt

    fig = completeness_violin(make_multiindex_df(), group_level="Condition")
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert sorted(labels) == ["Conditioned", "Fresh"]
    plt.close("all")


def test_completeness_violin_flat_df():
    """Flat columns collapse to a single group."""
    import matplotlib.pyplot as plt

    fig = completeness_violin(make_flat_df(), group_level=0)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["All samples"]
    plt.close("all")


def test_completeness_violin_known_values():
    """Means, labels, ordering and which groups get a violin body."""
    import matplotlib.pyplot as plt

    fig, data = completeness_violin(_violin_df(), group_level="Condition",
                                    return_data=True)
    ax = fig.axes[0]
    # A (mean 0.8) before B (mean 0.5)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]
    assert sorted(t.get_text() for t in ax.texts) == ["50.0%", "80.0%"]
    # B has zero variance: no KDE, but its points are still drawn
    assert len(_violin_bodies(ax)) == 1
    assert len(_scatter_points(ax)) == 2
    by_member = dict(zip(data["member"], data["value"]))
    assert by_member == pytest.approx(
        {"('A', 'A1')": 1.0, "('A', 'A2')": 0.8, "('A', 'A3')": 0.6,
         "('B', 'B1')": 0.5, "('B', 'B2')": 0.5})
    assert set(data["level"]) == {"samples"}
    plt.close("all")


def test_completeness_violin_level_features():
    """level="features" gives each feature's detection rate within the group."""
    import matplotlib.pyplot as plt

    fig, data = completeness_violin(_violin_df(), group_level="Condition",
                                    level="features", return_data=True)
    b = data[data["group"] == "B"].set_index("member")["value"]
    # every feature is missing in exactly one of B's two samples
    assert (b == 0.5).all() and len(b) == 10
    a = data[data["group"] == "A"].set_index("member")["value"]
    assert a["P0"] == pytest.approx(1 / 3) and a["P9"] == 1.0
    assert set(data["level"]) == {"features"}
    plt.close("all")


def test_completeness_violin_orders_groups_like_completeness_bars():
    """Groups are ordered by mean, as in completeness_bars. Here the median would
    put X first (1.0 against 0.8) but the mean puts Y first (0.8 against 0.7)."""
    import matplotlib.pyplot as plt

    data = np.ones((10, 6))
    data[:9, 2] = np.nan     # X: 1.0, 1.0, 0.1
    data[:2, 3:] = np.nan    # Y: 0.8, 0.8, 0.8
    cols = pd.MultiIndex.from_tuples(
        [("X", "x1"), ("X", "x2"), ("X", "x3"), ("Y", "y1"), ("Y", "y2"), ("Y", "y3")],
        names=["Condition", "Sample"])
    df = pd.DataFrame(data, columns=cols)
    bars = completeness_bars(df, "Condition", orientation="vertical")
    violin = completeness_violin(df, "Condition")
    order = [t.get_text() for t in bars.axes[0].get_xticklabels()]
    assert order == ["Y", "X"]
    assert [t.get_text() for t in violin.axes[0].get_xticklabels()] == order
    plt.close("all")


def test_completeness_violin_integer_group_level():
    """A level index works like its name, in the plot and in return_data."""
    import matplotlib.pyplot as plt

    fig_i, data_i = completeness_violin(_violin_df(), group_level=0, return_data=True)
    fig_n, data_n = completeness_violin(_violin_df(), group_level="Condition",
                                        return_data=True)
    assert ([t.get_text() for t in fig_i.axes[0].get_xticklabels()]
            == [t.get_text() for t in fig_n.axes[0].get_xticklabels()])
    pd.testing.assert_frame_equal(data_i, data_n)
    plt.close("all")


def test_completeness_violin_flat_return_data():
    """Flat columns are one group, keyed "all_samples" as in completeness_bars."""
    import matplotlib.pyplot as plt

    _, data = completeness_violin(make_flat_df(), group_level=0, return_data=True)
    assert set(data["group"]) == {"all_samples"}
    assert len(data) == 10
    plt.close("all")


@pytest.mark.parametrize("color, expected", [
    ("#123456", {"A": "#123456", "B": "#123456"}),
    ({"A": "#aa0000"}, {"A": "#aa0000", "B": "#4c72b0"}),  # unlisted -> default
])
def test_completeness_violin_colours(color, expected):
    """A single colour applies to every group; a dict maps groups and falls back
    to the default blue, as in completeness_bars."""
    from matplotlib.colors import to_hex

    import matplotlib.pyplot as plt

    fig = completeness_violin(_violin_df(), group_level="Condition", color=color)
    # scatter layers are drawn in group order: A, then B
    got = [to_hex(c.get_facecolor()[0]) for c in _scatter_points(fig.axes[0])]
    assert got == [expected["A"], expected["B"]]
    plt.close("all")


def test_completeness_violin_falls_back_to_vert_on_old_matplotlib(monkeypatch):
    """Before matplotlib 3.10 violinplot has no orientation= and takes vert=.
    The recording wrapper's own signature, (self, *args, **kwargs), has no
    orientation parameter, so it stands in for the old violinplot. It hands the
    real one orientation=, since current matplotlib deprecates vert=."""
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    calls = []
    original = Axes.violinplot

    def recording(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if "vert" in kwargs:
            kwargs["orientation"] = "vertical" if kwargs.pop("vert") else "horizontal"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "violinplot", recording)
    completeness_violin(_violin_df(), group_level="Condition", orientation="horizontal")
    assert len(calls) == 1
    assert calls[0].get("vert") is False and "orientation" not in calls[0]
    plt.close("all")


def test_completeness_violin_threshold():
    """The threshold is a horizontal line when vertical."""
    import matplotlib.pyplot as plt

    fig = completeness_violin(make_multiindex_df(), group_level="Condition",
                              threshold=0.7)
    ys = [ln.get_ydata() for ln in fig.axes[0].get_lines()
          if ln.get_label() == "70% threshold"]
    assert len(ys) == 1 and list(ys[0]) == [0.7, 0.7]
    plt.close("all")


def test_completeness_violin_horizontal():
    """Horizontal puts groups on the y-axis and the threshold on x."""
    import matplotlib.pyplot as plt

    fig = completeness_violin(make_multiindex_df(), group_level="Condition",
                              orientation="horizontal", threshold=0.7)
    ax = fig.axes[0]
    assert sorted(t.get_text() for t in ax.get_yticklabels()) == ["Conditioned", "Fresh"]
    xs = [ln.get_xdata() for ln in ax.get_lines() if ln.get_label() == "70% threshold"]
    assert len(xs) == 1 and list(xs[0]) == [0.7, 0.7]
    plt.close("all")


def test_completeness_violin_no_matplotlib_deprecation_warnings():
    """violinplot's vert= is deprecated from matplotlib 3.10; neither orientation
    may warn."""
    import warnings

    import matplotlib.pyplot as plt

    for orientation in ("vertical", "horizontal"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", PendingDeprecationWarning)
            warnings.simplefilter("error", DeprecationWarning)
            completeness_violin(make_multiindex_df(), group_level="Condition",
                                orientation=orientation)
    plt.close("all")


def test_completeness_violin_hide_points():
    import matplotlib.pyplot as plt

    fig = completeness_violin(_violin_df(), group_level="Condition", show_points=False)
    assert _scatter_points(fig.axes[0]) == []
    plt.close("all")


@pytest.mark.parametrize("kwargs", [{"level": "sample"}, {"orientation": "vert"}])
def test_completeness_violin_rejects_unknown_options(kwargs):
    with pytest.raises(ValueError):
        completeness_violin(_violin_df(), group_level="Condition", **kwargs)


def test_completeness_violin_zero_variance():
    """All-present groups have no spread (ptp == 0): skip the violin, not crash."""
    import matplotlib.pyplot as plt

    df = make_multiindex_df().fillna(1.0)
    fig = completeness_violin(df, group_level="Condition")
    assert _violin_bodies(fig.axes[0]) == []
    plt.close("all")


def test_completeness_violin_all_missing():
    import matplotlib.pyplot as plt

    df = make_multiindex_df()
    df.loc[:, :] = np.nan
    fig = completeness_violin(df, group_level="Condition")
    assert sorted(t.get_text() for t in fig.axes[0].texts) == ["0.0%", "0.0%"]
    plt.close("all")


def test_completeness_violin_all_present():
    import matplotlib.pyplot as plt

    df = make_multiindex_df().fillna(1.0)
    fig = completeness_violin(df, group_level="Condition")
    assert sorted(t.get_text() for t in fig.axes[0].texts) == ["100.0%", "100.0%"]
    plt.close("all")


def test_completeness_violin_save_to_disk(tmp_path: Path):
    """Save parameter writes a PNG file."""
    import matplotlib.pyplot as plt

    out = tmp_path / "violin.png"
    completeness_violin(make_multiindex_df(), group_level="Condition",
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
