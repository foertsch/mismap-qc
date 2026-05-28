"""Tests for estimate_lod()."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from mismap_qc import estimate_lod  # noqa: E402


def _df(rows):
    return pd.DataFrame(
        rows,
        index=[f"F{i}" for i in range(len(rows))],
        columns=[f"S{i}" for i in range(len(rows[0]))],
    )


def test_min_recovers_known_minima():
    df = _df([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, np.nan, 30.0, 40.0],
        [100.0, 200.0, 300.0, 400.0],
    ])
    lod = estimate_lod(df, method="min")
    assert lod.loc["F0"] == 1.0
    assert lod.loc["F1"] == 10.0
    assert lod.loc["F2"] == 100.0


def test_quantile_method():
    df = _df([[1.0, 2.0, 3.0, 4.0, 5.0]])
    lod = estimate_lod(df, method="quantile", quantile=0.0)
    assert lod.loc["F0"] == 1.0  # 0th quantile == min
    lod50 = estimate_lod(df, method="quantile", quantile=0.5)
    assert lod50.loc["F0"] == 3.0  # median


def test_min_present_threshold():
    df = _df([
        [1.0, 2.0, 3.0, 4.0],         # 4 obs
        [10.0, 20.0, np.nan, np.nan], # 2 obs
        [np.nan, np.nan, np.nan, 99], # 1 obs
    ])
    lod = estimate_lod(df, min_present=3)
    assert lod.loc["F0"] == 1.0
    assert np.isnan(lod.loc["F1"])
    assert np.isnan(lod.loc["F2"])


def test_all_missing_feature_returns_nan():
    df = _df([
        [1.0, 2.0, 3.0, 4.0],
        [np.nan, np.nan, np.nan, np.nan],
    ])
    lod = estimate_lod(df)
    assert lod.loc["F0"] == 1.0
    assert np.isnan(lod.loc["F1"])


def test_returns_series_with_lod_name():
    df = _df([[1.0, 2.0]])
    lod = estimate_lod(df, min_present=2)
    assert isinstance(lod, pd.Series)
    assert lod.name == "lod"


def test_invalid_method_raises():
    df = _df([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="method must be"):
        estimate_lod(df, method="median")


def test_invalid_quantile_raises():
    df = _df([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="quantile must be"):
        estimate_lod(df, method="quantile", quantile=1.5)
