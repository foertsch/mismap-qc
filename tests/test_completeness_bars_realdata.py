"""Real-data tests for completeness_bars() using CPTAC LUAD proteomics.

These tests are skipped automatically when the data files are not present,
so CI stays green without committing large data files to the repo.
Data lives in examples/data/ (gitignored).
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))
from mismap_qc import completeness_bars


def _make_multiindex_df(cptac_df):
    """Attach a Tumor_Normal MultiIndex level to the CPTAC combined matrix."""
    tumor_normal = ["Tumor" if c.endswith("_T") else "Normal"
                    for c in cptac_df.columns]
    columns = pd.MultiIndex.from_arrays(
        [tumor_normal, cptac_df.columns.tolist()],
        names=["Tumor_Normal", "Sample"],
    )
    return cptac_df.set_axis(columns, axis=1)


def test_completeness_bars_real_data_returns_figure(cptac_df):
    if cptac_df is None:
        pytest.skip("CPTAC data not available")
    df = _make_multiindex_df(cptac_df)
    fig = completeness_bars(df, group_level="Tumor_Normal")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_real_data_threshold(cptac_df):
    if cptac_df is None:
        pytest.skip("CPTAC data not available")
    df = _make_multiindex_df(cptac_df)
    fig = completeness_bars(df, group_level="Tumor_Normal", threshold=0.7)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_real_data_vertical(cptac_df):
    if cptac_df is None:
        pytest.skip("CPTAC data not available")
    df = _make_multiindex_df(cptac_df)
    fig = completeness_bars(df, group_level="Tumor_Normal",
                            orientation="vertical")
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_completeness_bars_real_data_save(cptac_df, tmp_path):
    if cptac_df is None:
        pytest.skip("CPTAC data not available")
    df = _make_multiindex_df(cptac_df)
    out = tmp_path / "completeness_real.png"
    completeness_bars(df, group_level="Tumor_Normal", save=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close("all")
