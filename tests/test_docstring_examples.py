"""Execute the ``Examples`` block of every public function.

pyOpenSci's submission checklist asks for documentation with examples for all
functions, and the documentation site renders these docstrings directly. An
example that no longer runs is worse than no example, so every ``>>>`` line in a
public docstring is executed here against a shared fixture.

Examples are written to use a DataFrame named ``df`` with a MultiIndex column
index carrying ``Batch`` and ``Condition`` levels, plus ``adata`` for the AnnData
reader. Keep new examples to those names, or extend the namespace built in
``test_docstring_examples_run``.
"""
from __future__ import annotations

import re

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import mismap_qc  # noqa: E402

# Functions whose examples need an optional dependency, and the module to skip on.
_OPTIONAL = {
    "missing_matrix_html": "plotly",
    "missing_upset": "upsetplot",
    "from_anndata": "anndata",
}

PUBLIC = [
    "qc",
    "assert_qc",
    "missing_matrix",
    "missing_matrix_html",
    "missing_abundance_density",
    "completeness_bars",
    "detection_waterfall",
    "missing_runorder",
    "missing_mechanism",
    "comissing_heatmap",
    "missing_upset",
    "from_anndata",
    "estimate_lod",
]


def _example_df() -> pd.DataFrame:
    """Features x samples with structured missingness and two annotation levels."""
    rng = np.random.default_rng(0)
    n_features, n_per_group = 60, 5
    columns = pd.MultiIndex.from_tuples(
        [
            (batch, condition, f"{condition}_{batch}_{i}")
            for batch in ("B1", "B2")
            for condition in ("Control", "Treated")
            for i in range(n_per_group)
        ],
        names=["Batch", "Condition", "Sample"],
    )
    data = rng.lognormal(mean=3.0, sigma=1.0, size=(n_features, len(columns)))
    df = pd.DataFrame(data, index=[f"P{i:03d}" for i in range(n_features)], columns=columns)

    # Low-abundance features drop out (MNAR), plus a scatter of random dropout (MAR)
    # and one clearly worse sample, so outlier and mechanism checks have something
    # to find rather than returning empty frames.
    df.iloc[:12] = df.iloc[:12].mask(rng.random(df.iloc[:12].shape) < 0.6)
    df = df.mask(rng.random(df.shape) < 0.05)
    df.iloc[:, 3] = df.iloc[:, 3].mask(rng.random(n_features) < 0.5)
    return df


def _extract_examples(func) -> list[str]:
    doc = func.__doc__ or ""
    match = re.search(r"Examples\s*\n\s*-+\s*\n(.*?)(?:\n\s*\n|\Z)", doc, re.S)
    if not match:
        return []
    return [
        line.strip()[4:]
        for line in match.group(1).splitlines()
        if line.strip().startswith(">>> ")
    ]


@pytest.mark.parametrize("name", PUBLIC)
def test_public_function_has_examples(name):
    """Every public function documents at least one example."""
    assert _extract_examples(getattr(mismap_qc, name)), f"{name} has no Examples block"


@pytest.mark.parametrize("name", PUBLIC)
def test_docstring_examples_run(name):
    """Every documented example executes without raising."""
    dep = _OPTIONAL.get(name)
    if dep:
        pytest.importorskip(dep, reason=f"{name} examples need {dep}")

    ns = {n: getattr(mismap_qc, n) for n in PUBLIC}
    ns.update(df=_example_df(), pd=pd, np=np)

    if name == "from_anndata":
        import anndata

        base = _example_df()
        ns["adata"] = anndata.AnnData(
            X=base.to_numpy().T,
            obs=pd.DataFrame(
                {
                    "Batch": base.columns.get_level_values("Batch"),
                    "Condition": base.columns.get_level_values("Condition"),
                },
                index=base.columns.get_level_values("Sample"),
            ),
            var=pd.DataFrame({"gene_symbol": base.index}, index=base.index),
        )

    for line in _extract_examples(getattr(mismap_qc, name)):
        try:
            exec(compile(line, f"<{name} example>", "single"), ns)  # noqa: S102
        except Exception as exc:  # pragma: no cover - failure path is the point
            pytest.fail(f"example failed for {name}:\n    >>> {line}\n  {exc!r}")
        finally:
            plt.close("all")
