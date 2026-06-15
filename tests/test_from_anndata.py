"""Tests for from_anndata() interop reader."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from mismap_qc import from_anndata  # noqa: E402

anndata = pytest.importorskip("anndata")


def _make_adata(n_obs=12, n_vars=20, missing_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(10, 2, (n_obs, n_vars))
    mask = rng.random((n_obs, n_vars)) < missing_frac
    X[mask] = np.nan
    obs = pd.DataFrame(
        {
            "batch": ["B1"] * (n_obs // 2) + ["B2"] * (n_obs - n_obs // 2),
            "condition": (["ctrl", "treat"] * ((n_obs + 1) // 2))[:n_obs],
        },
        index=[f"S{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_symbol": [f"sym_{i}" for i in range(n_vars)]},
        index=[f"F{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=X, obs=obs, var=var)


def test_basic_conversion():
    adata = _make_adata()
    df = from_anndata(adata)
    assert df.shape == (20, 12)  # features x samples (transposed)
    # Default: flat Index of sample names
    assert not isinstance(df.columns, pd.MultiIndex)


def test_obs_levels_builds_multiindex():
    adata = _make_adata()
    df = from_anndata(adata, obs_levels=["batch", "condition"])
    assert isinstance(df.columns, pd.MultiIndex)
    assert list(df.columns.names) == ["batch", "condition", "sample"]
    assert df.shape == (20, 12)


def test_var_index_uses_alt_feature_name():
    adata = _make_adata()
    df = from_anndata(adata, var_index="gene_symbol")
    assert list(df.index)[:3] == ["sym_0", "sym_1", "sym_2"]


def test_missing_value_zero():
    rng = np.random.default_rng(0)
    X = rng.random((5, 10))
    X[X < 0.3] = 0  # ~30% zeros
    adata = anndata.AnnData(X=X)
    df = from_anndata(adata, missing_value="zero")
    assert df.isna().sum().sum() > 0
    # No zeros remain
    assert (df.fillna(1) > 0).all().all()


def test_missing_value_threshold():
    X = np.array([[0.5, 1.5, 2.5], [0.1, 5.0, 0.8]])
    adata = anndata.AnnData(X=X)
    df = from_anndata(adata, missing_value=1.0)
    # Values <= 1.0 become NaN
    # df is 3 features x 2 samples (transposed)
    arr = df.values
    assert np.isnan(arr).sum() == 3  # 0.5, 0.1, 0.8 are below threshold


def test_layer_selection():
    rng = np.random.default_rng(0)
    X = rng.normal(10, 2, (4, 6))
    layer = rng.normal(20, 2, (4, 6))
    adata = anndata.AnnData(X=X, layers={"norm": layer})
    df_x = from_anndata(adata)
    df_l = from_anndata(adata, layer="norm")
    assert not np.allclose(df_x.values, df_l.values)


def test_unknown_layer_raises():
    adata = _make_adata()
    with pytest.raises(ValueError, match="Layer 'missing' not in adata.layers"):
        from_anndata(adata, layer="missing")


def test_unknown_obs_levels_raises():
    adata = _make_adata()
    with pytest.raises(ValueError, match="obs_levels not present"):
        from_anndata(adata, obs_levels=["nonexistent"])


def test_unknown_var_index_raises():
    adata = _make_adata()
    with pytest.raises(ValueError, match="var_index"):
        from_anndata(adata, var_index="nonexistent")


def test_invalid_missing_value_raises():
    adata = _make_adata()
    with pytest.raises(ValueError, match="missing_value must be"):
        from_anndata(adata, missing_value="invalid")


def test_non_anndata_raises():
    with pytest.raises(TypeError, match="Expected AnnData"):
        from_anndata({"X": np.zeros((3, 3))})


def test_sparse_input_densified():
    sparse = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(0)
    X = rng.random((5, 10))
    X[X < 0.3] = 0
    X_sparse = sparse.csr_matrix(X)
    adata = anndata.AnnData(X=X_sparse)
    df = from_anndata(adata)
    assert df.shape == (10, 5)
    assert isinstance(df, pd.DataFrame)


def test_roundtrip_into_qc():
    """End-to-end: AnnData -> from_anndata -> qc()."""
    from mismap_qc import qc

    adata = _make_adata()
    df = from_anndata(adata, obs_levels=["condition"])
    report = qc(df, group_level="condition")
    assert report.passed in (True, False)  # whatever it is, it didn't crash
    assert report.n_features == 20
    assert report.n_samples == 12
