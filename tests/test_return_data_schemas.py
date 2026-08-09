"""Schema regression test for the return_data=True flag on every plot function.

If you rename or remove a column, this test fails. Adding a column is allowed
(the test checks that expected columns are present, not that they are the only
columns). To rename: change the schema in mismap_qc._RETURN_DATA_SCHEMAS and
update this test in the same commit, plus document the change in CHANGELOG.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))
from mismap_qc import (  # noqa: E402
    _RETURN_DATA_SCHEMAS,
    comissing_heatmap,
    completeness_bars,
    detection_waterfall,
    missing_matrix,
    missing_runorder,
    missing_upset,
)


def _make_df(n_features=20, n_samples=10, missing_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(10, 2, (n_features, n_samples))
    mask = rng.random((n_features, n_samples)) < missing_frac
    data[mask] = np.nan
    return pd.DataFrame(
        data,
        index=[f"F{i}" for i in range(n_features)],
        columns=[f"S{i}" for i in range(n_samples)],
    )


def _make_multi(n_features=20, n_per_group=5, missing_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    tuples = [(c, f"S{c}{i}") for c in ["A", "B"] for i in range(n_per_group)]
    cols = pd.MultiIndex.from_tuples(tuples, names=["condition", "sample"])
    data = rng.normal(10, 2, (n_features, len(tuples)))
    mask = rng.random((n_features, len(tuples))) < missing_frac
    data[mask] = np.nan
    return pd.DataFrame(data, index=[f"F{i}" for i in range(n_features)], columns=cols)


# Registry of (plot_function, kwargs, schema_key)
_CASES = [
    (missing_matrix, {}, "missing_matrix"),
    (completeness_bars, {"group_level": "condition"}, "completeness_bars"),
    (detection_waterfall, {}, "detection_waterfall"),
    (missing_runorder, {}, "missing_runorder"),
    (comissing_heatmap, {"top_n": 10}, "comissing_heatmap"),
    (missing_upset, {}, "missing_upset"),
]

# missing_upset is the only plot behind an optional dependency.
_NEEDS_EXTRA = {"missing_upset": "upsetplot"}


def _skip_without_extra(plot_fn):
    extra = _NEEDS_EXTRA.get(plot_fn.__name__)
    if extra:
        pytest.importorskip(extra, reason=f"{plot_fn.__name__} needs {extra}")


@pytest.mark.parametrize("plot_fn,kwargs,schema_key", _CASES)
def test_return_data_returns_tuple(plot_fn, kwargs, schema_key):
    import matplotlib.pyplot as plt

    _skip_without_extra(plot_fn)

    df = _make_multi() if "group_level" in kwargs else _make_df()
    result = plot_fn(df, return_data=True, **kwargs)
    assert isinstance(result, tuple), (
        f"{plot_fn.__name__} with return_data=True should return a tuple"
    )
    assert len(result) == 2
    fig, data = result
    assert isinstance(fig, plt.Figure)
    assert isinstance(data, pd.DataFrame)
    plt.close("all")


@pytest.mark.parametrize("plot_fn,kwargs,schema_key", _CASES)
def test_return_data_schema_columns(plot_fn, kwargs, schema_key):
    import matplotlib.pyplot as plt

    _skip_without_extra(plot_fn)
    df = _make_multi() if "group_level" in kwargs else _make_df()
    _, data = plot_fn(df, return_data=True, **kwargs)
    expected = _RETURN_DATA_SCHEMAS[schema_key]
    missing_cols = [c for c in expected if c not in data.columns]
    assert not missing_cols, (
        f"{plot_fn.__name__} return_data is missing columns: {missing_cols}. "
        f"Got: {list(data.columns)}"
    )
    plt.close("all")


@pytest.mark.parametrize("plot_fn,kwargs,schema_key", _CASES)
def test_return_data_default_is_figure_only(plot_fn, kwargs, schema_key):
    """Default behavior (no return_data flag) returns just a Figure."""
    import matplotlib.pyplot as plt

    _skip_without_extra(plot_fn)
    df = _make_multi() if "group_level" in kwargs else _make_df()
    result = plot_fn(df, **kwargs)
    assert isinstance(result, plt.Figure)
    plt.close("all")


def test_missing_matrix_split_by_with_return_data():
    """missing_matrix's split_by branch should still honor return_data."""
    import matplotlib.pyplot as plt

    df = _make_multi()
    result = missing_matrix(df, split_by="condition", return_data=True)
    assert isinstance(result, tuple)
    fig, data = result
    assert isinstance(fig, plt.Figure)
    expected = _RETURN_DATA_SCHEMAS["missing_matrix"]
    for c in expected:
        assert c in data.columns
    plt.close("all")


def test_schemas_registry_covers_all_plots():
    """If a plot is added but forgotten in the schemas registry, fail loudly."""
    plot_names = {fn.__name__ for fn, _, _ in _CASES}
    assert set(_RETURN_DATA_SCHEMAS) == plot_names
