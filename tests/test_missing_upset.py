"""Tests for missing_upset() and the intersection computation behind it.

The plot needs upsetplot and skips without it. The intersection logic in
_upset_intersections() is pure pandas/numpy and always runs, which is where the
decisions worth guarding live: the 50% group threshold, the intersection cap, and
the promise that capped intersections are still returned.
"""
from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mismap_qc import missing_upset  # noqa: E402
from mismap_qc.stats import _upset_intersections  # noqa: E402

upsetplot = pytest.importorskip("upsetplot", reason="missing_upset needs the [upset] extra")

SCHEMA = ["feature", "members", "n_features", "rank", "plotted"]


def make_flat_df():
    """Four features, three samples, two distinct co-missingness patterns."""
    return pd.DataFrame(
        {
            "S1": [np.nan, 1.0, np.nan, 1.0],
            "S2": [np.nan, 1.0, 1.0, 1.0],
            "S3": [1.0, 1.0, np.nan, 1.0],
        },
        index=["F1", "F2", "F3", "F4"],
    )


def make_multiindex_df():
    cols = pd.MultiIndex.from_tuples(
        [("ctrl", "S1"), ("ctrl", "S2"), ("treat", "S3"), ("treat", "S4")],
        names=["condition", "sample"],
    )
    values = [
        [np.nan, np.nan, 1.0, 1.0],   # lost in the whole ctrl group
        [np.nan, 1.0, 1.0, 1.0],      # lost in one ctrl sample only
        [1.0, 1.0, 1.0, 1.0],         # detected everywhere
        [np.nan, np.nan, np.nan, 1.0],
    ]
    return pd.DataFrame(values, index=["F1", "F2", "F3", "F4"], columns=cols)


# --- figure behaviour -------------------------------------------------------


def test_missing_upset_returns_figure():
    fig = missing_upset(make_flat_df())
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_upset_multiindex():
    fig = missing_upset(make_multiindex_df())
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_missing_upset_group_mode():
    fig, table = missing_upset(make_multiindex_df(), by="condition", return_data=True)
    assert isinstance(fig, plt.Figure)
    # F1 is missing in both ctrl samples, so it is missing in the ctrl group.
    assert table.loc[table.feature == "F1", "members"].iloc[0] == "ctrl"
    plt.close("all")


def test_missing_upset_all_missing():
    df = pd.DataFrame(np.nan, index=["F1", "F2"], columns=["S1", "S2"])
    fig, table = missing_upset(df, return_data=True)
    assert isinstance(fig, plt.Figure)
    # One intersection: everything missing everywhere.
    assert table["rank"].nunique() == 1
    assert table["members"].unique().tolist() == ["S1|S2"]
    plt.close("all")


def test_missing_upset_all_present():
    df = pd.DataFrame(1.0, index=["F1", "F2"], columns=["S1", "S2"])
    fig, table = missing_upset(df, return_data=True)
    assert isinstance(fig, plt.Figure)
    assert table.empty
    assert list(table.columns) == SCHEMA
    plt.close("all")


def test_missing_upset_save_to_disk(tmp_path):
    out = tmp_path / "upset.png"
    missing_upset(make_flat_df(), save=str(out))
    assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_missing_upset_return_data_schema():
    fig, table = missing_upset(make_flat_df(), return_data=True)
    assert list(table.columns) == SCHEMA
    assert len(table) == 2  # F2 and F4 are fully detected and excluded
    plt.close("all")


def test_missing_upset_emits_no_futurewarnings():
    """upsetplot 0.9.0 emits pandas FutureWarnings from its own internals. They are
    suppressed narrowly around the plot call so callers are not shown warnings they
    cannot act on."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        missing_upset(make_flat_df())
    leaked = [str(w.message) for w in caught if issubclass(w.category, FutureWarning)]
    assert not leaked, f"leaked upstream FutureWarnings: {leaked}"
    plt.close("all")


# --- intersection logic -----------------------------------------------------


def test_intersections_exclude_fully_detected_features():
    table = _upset_intersections(make_flat_df())
    assert set(table.feature) == {"F1", "F3"}


def test_intersections_are_ranked_by_size():
    # F1, F2 share a pattern (2 features); F3 is alone (1 feature).
    df = pd.DataFrame(
        {
            "S1": [np.nan, np.nan, np.nan],
            "S2": [np.nan, np.nan, 1.0],
        },
        index=["F1", "F2", "F3"],
    )
    table = _upset_intersections(df)
    biggest = table[table["rank"] == 1]
    assert set(biggest.feature) == {"F1", "F2"}
    assert biggest["n_features"].unique().tolist() == [2]


def test_cap_marks_rather_than_drops_intersections():
    """max_intersections limits what is drawn, not what is returned."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=(200, 8))
    values[rng.random((200, 8)) < 0.25] = np.nan
    df = pd.DataFrame(
        values,
        index=[f"P{i}" for i in range(200)],
        columns=[f"S{j}" for j in range(8)],
    )
    table = _upset_intersections(df, max_intersections=5)
    assert table["rank"].nunique() > 5, "test data should exceed the cap"
    assert table.loc[table["plotted"], "rank"].nunique() == 5
    assert (table.loc[table["plotted"], "rank"] <= 5).all()
    assert not table.loc[~table["plotted"], "rank"].empty


def test_min_size_filters_small_intersections():
    df = pd.DataFrame(
        {
            "S1": [np.nan, np.nan, np.nan],
            "S2": [np.nan, np.nan, 1.0],
        },
        index=["F1", "F2", "F3"],
    )
    table = _upset_intersections(df, min_size=2).set_index("feature")
    assert not table.loc["F3", "plotted"]  # its intersection holds only one feature
    assert table.loc["F1", "plotted"]


def test_group_min_frac_threshold():
    """Default 0.5: a feature counts as missing in a group once the majority of
    that group's samples are missing it."""
    cols = pd.MultiIndex.from_tuples(
        [("g", "S1"), ("g", "S2"), ("g", "S3"), ("h", "S4")],
        names=["grp", "sample"],
    )
    # F1 missing in 1 of 3 g-samples (below 0.5), F2 missing in 2 of 3 (above).
    df = pd.DataFrame(
        [[np.nan, 1.0, 1.0, 1.0], [np.nan, np.nan, 1.0, 1.0]],
        index=["F1", "F2"],
        columns=cols,
    )
    table = _upset_intersections(df, by="grp")
    assert "F1" not in set(table.feature)
    assert table.loc[table.feature == "F2", "members"].iloc[0] == "g"

    # Lowering the threshold brings F1 in.
    loosened = _upset_intersections(df, by="grp", group_min_frac=0.3)
    assert "F1" in set(loosened.feature)


def test_group_mode_requires_multiindex():
    with pytest.raises(ValueError, match="MultiIndex"):
        _upset_intersections(make_flat_df(), by="condition")


def test_members_are_pipe_joined_and_actionable():
    """The documented use: go from an intersection straight to its feature list."""
    table = _upset_intersections(make_flat_df())
    features = table.query("members == 'S1|S2'").feature.tolist()
    assert features == ["F1"]
