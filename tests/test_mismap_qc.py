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
    completeness_bars,
    detection_waterfall,
    missing_matrix,
    missing_matrix_html,
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
