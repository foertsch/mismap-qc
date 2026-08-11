"""Scope E.1: estimate_lod() per-feature limit-of-detection estimate."""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_lod(
    df: pd.DataFrame,
    *,
    method: str = "min",
    quantile: float = 0.05,
    min_present: int = 3,
) -> pd.Series:
    """Estimate a per-feature limit of detection from observed values.

    For each feature, returns either the minimum observed value (``method="min"``)
    or a low-quantile observed value (``method="quantile"``). Features with
    fewer than ``min_present`` observations return ``NaN`` (cannot estimate).

    Parameters
    ----------
    df : DataFrame
        features (rows) x samples (columns). NaN = missing.
    method : str
        ``"min"`` (default) or ``"quantile"``.
    quantile : float
        Quantile in [0, 1] used when ``method="quantile"``. Default 0.05.
    min_present : int
        Minimum non-NaN observations required to estimate. Features below
        this threshold return NaN.

    Returns
    -------
    pandas.Series
        Per-feature LOD estimate, indexed by feature name.

    Raises
    ------
    ValueError
        If ``method`` is not one of {"min", "quantile"} or ``quantile`` is
        outside [0, 1].

    Examples
    --------
    >>> lod = estimate_lod(df)
    >>> lod = estimate_lod(df, method="quantile", quantile=0.1)
"""
    if method not in ("min", "quantile"):
        raise ValueError(
            f"method must be 'min' or 'quantile'; got {method!r}"
        )
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1]; got {quantile}")

    n_present = df.notna().sum(axis=1)
    if method == "min":
        lod = df.min(axis=1, skipna=True)
    else:
        # Per-feature quantile of non-NaN values
        lod = df.quantile(quantile, axis=1, numeric_only=True)
    lod[n_present < min_present] = np.nan
    return lod.rename("lod")
