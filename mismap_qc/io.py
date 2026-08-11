"""Interop readers: from_anndata()."""
from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd


def from_anndata(
    adata,
    *,
    layer: str | None = None,
    missing_value: float | str = "nan",
    obs_levels: list[str] | None = None,
    var_index: str | None = None,
    transpose: bool = True,
) -> pd.DataFrame:
    """Convert an AnnData object to the features x samples DataFrame mismap-qc expects.

    AnnData stores data as obs (rows, typically samples or cells) x var (columns,
    typically features). mismap-qc wants the inverse. With ``transpose=True`` (the
    default), the output is features x samples.

    Parameters
    ----------
    adata : AnnData
        Input. Requires anndata to be installed (``pip install mismap-qc[anndata]``).
    layer : str, optional
        Layer name to use. None => adata.X.
    missing_value : float or str
        How to treat missing values:
          - ``"nan"`` (default): keep NaN as missing.
          - ``"zero"``: treat exact zeros as missing.
          - float: treat values below this threshold as missing (useful for
            log-intensity matrices with a noise floor).
    obs_levels : list of str, optional
        obs columns to use as additional MultiIndex levels on the sample axis.
        Example: ``obs_levels=["batch", "condition"]`` produces a 3-level
        MultiIndex with levels [batch, condition, sample].
    var_index : str, optional
        var column to use as feature names. None => adata.var_names.
    transpose : bool
        If True (default), output is features x samples. Set False if your
        AnnData is already in features x samples orientation (rare).

    Returns
    -------
    DataFrame
        features (rows) x samples (columns). When ``obs_levels`` is set,
        columns are a MultiIndex with the obs levels and the sample name as
        the innermost level.

    Raises
    ------
    ImportError
        If anndata is not installed.
    ValueError
        If ``layer``, ``obs_levels``, ``var_index``, or ``missing_value`` is
        invalid.

    Examples
    --------
    >>> df = from_anndata(adata)
    >>> df = from_anndata(adata, obs_levels=["Batch", "Condition"])
    >>> df = from_anndata(adata, missing_value=0.0, var_index="gene_symbol")
"""
    try:
        import anndata as _anndata
    except ImportError as e:
        raise ImportError(
            "from_anndata requires anndata. Install with: "
            "pip install mismap-qc[anndata]"
        ) from e

    if not isinstance(adata, _anndata.AnnData):
        raise TypeError(
            f"Expected AnnData object, got {type(adata).__name__}"
        )

    # Validate obs_levels early to avoid wasted work on big objects
    if obs_levels is not None:
        bad = [c for c in obs_levels if c not in adata.obs.columns]
        if bad:
            raise ValueError(
                f"obs_levels not present in adata.obs: {bad}. "
                f"Available: {list(adata.obs.columns)}"
            )

    if var_index is not None and var_index not in adata.var.columns:
        raise ValueError(
            f"var_index {var_index!r} not in adata.var.columns. "
            f"Available: {list(adata.var.columns)}"
        )

    # Pick the data matrix
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise ValueError(
                f"Layer {layer!r} not in adata.layers. "
                f"Available: {list(adata.layers)}"
            )
        X = adata.layers[layer]

    # Densify if sparse
    try:
        from scipy import sparse as _sparse
        if _sparse.issparse(X):
            estimated_bytes = X.shape[0] * X.shape[1] * 8
            if estimated_bytes > 1e9:
                _warnings.warn(
                    f"Densifying a large sparse matrix "
                    f"(~{estimated_bytes / 1e9:.1f} GB).",
                    UserWarning,
                )
            X = X.toarray()
    except ImportError:
        pass

    X = np.asarray(X, dtype=float)

    # Feature / sample names
    if var_index is None:
        feature_names = list(adata.var_names)
    else:
        feature_names = list(adata.var[var_index].astype(str))

    sample_names = list(adata.obs_names)

    # Build column index
    if obs_levels:
        levels_data = [list(adata.obs[c]) for c in obs_levels]
        tuples = list(zip(*levels_data, sample_names))
        columns = pd.MultiIndex.from_tuples(tuples, names=list(obs_levels) + ["sample"])
    else:
        columns = pd.Index(sample_names, name="sample")

    # Build DataFrame
    if transpose:
        # X is obs x var; we want features (var) x samples (obs)
        df = pd.DataFrame(X.T, index=feature_names, columns=columns)
    else:
        df = pd.DataFrame(X, index=feature_names, columns=columns)

    # Apply missing_value strategy
    if missing_value == "nan":
        pass  # already NaN-encoded
    elif missing_value == "zero":
        df = df.where(df != 0)
    elif isinstance(missing_value, (int, float)) and not isinstance(missing_value, bool):
        threshold = float(missing_value)
        df = df.where(df > threshold)
    else:
        raise ValueError(
            f"missing_value must be 'nan', 'zero', or a float threshold; "
            f"got {missing_value!r}"
        )

    return df
