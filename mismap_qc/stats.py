"""Pure-numeric analytical helpers shared by qc() and the plot functions.

None of these produce figures. Each operates on a features x samples DataFrame
(or a derived structure) and returns a DataFrame, Series, or dict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _resolve_group_labels(df: pd.DataFrame, group_level):
    """Return (level_index, group_array) or (None, None) when group_level is None."""
    if group_level is None:
        return None, None
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            "group_level was specified but df.columns is not a MultiIndex."
        )
    if isinstance(group_level, str):
        lv = list(df.columns.names).index(group_level)
    else:
        lv = group_level
    groups = np.array([t[lv] for t in df.columns])
    return lv, groups

def _compute_sample_outliers(df, sample_missing_rate, groups, *, z_threshold=2.5):
    """Per-sample z-score of missingness, within group if groups provided."""
    miss = sample_missing_rate.values.astype(float)
    sample_names = [str(c) for c in df.columns]
    z = np.zeros_like(miss)
    if groups is not None:
        group_labels = list(groups)
        unique = np.unique(groups)
        for g in unique:
            mask = groups == g
            if mask.sum() >= 3:
                sub = miss[mask]
                std = float(np.std(sub, ddof=1)) if mask.sum() > 1 else 0.0
                if std > 0:
                    z[mask] = (sub - sub.mean()) / std
    else:
        group_labels = [None] * len(miss)
        std = float(np.std(miss, ddof=1)) if len(miss) > 1 else 0.0
        if std > 0:
            z = (miss - miss.mean()) / std
    return pd.DataFrame(
        {
            "sample": sample_names,
            "group": group_labels,
            "missing_rate": miss,
            "z_score": z,
            "flagged": np.abs(z) > z_threshold,
        }
    )

def _classify_mechanism(df: pd.DataFrame, *, min_present: int = 3, alpha: float = 0.05) -> pd.DataFrame:
    """Per-feature MNAR / MAR classification via Mann-Whitney U.

    Per-sample mean abundance is computed once. For each feature, compare the
    sample-mean abundance distribution of (samples where feature is detected)
    vs (samples where feature is missing). One-sided test for present_means
    > absent_means: significant => MNAR.

    Returns DataFrame: feature, mechanism, missing_rate, mean_abundance, p_value.
    """
    from scipy import stats

    sample_mean = df.mean(axis=0, skipna=True).values
    rows = []
    for feature, vals in df.iterrows():
        present_mask = vals.notna().values
        n_present = int(present_mask.sum())
        n_total = len(vals)
        missing_rate = 1.0 - (n_present / n_total)
        mean_abundance = float(np.nanmean(vals)) if n_present else np.nan
        if n_present < min_present or (n_total - n_present) < min_present:
            rows.append((feature, "INSUFFICIENT", missing_rate, mean_abundance, np.nan))
            continue
        present_means = sample_mean[present_mask]
        absent_means = sample_mean[~present_mask]
        try:
            _, p = stats.mannwhitneyu(
                present_means, absent_means, alternative="greater"
            )
        except ValueError:
            p = np.nan
        if np.isnan(p):
            mech = "INSUFFICIENT"
        elif p < alpha:
            mech = "MNAR"
        else:
            mech = "MAR"
        rows.append((feature, mech, missing_rate, mean_abundance, float(p)))
    return pd.DataFrame(
        rows, columns=["feature", "mechanism", "missing_rate", "mean_abundance", "p_value"]
    )

def _top_codropouts(df: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    """Top co-missing feature pairs by fraction-of-samples-co-missing."""
    M = df.isna().astype(np.int8).values
    has_missing = M.sum(axis=1) > 0
    if has_missing.sum() < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "comissingness"])
    Mf = M[has_missing]
    names = df.index[has_missing].tolist()
    n_samples = M.shape[1]
    co = (Mf @ Mf.T) / n_samples
    n = co.shape[0]
    iu = np.triu_indices(n, k=1)
    co_vals = co[iu]
    if len(co_vals) == 0:
        return pd.DataFrame(columns=["feature_a", "feature_b", "comissingness"])
    order = np.argsort(co_vals)[::-1][:top_n]
    rows = [(names[iu[0][i]], names[iu[1][i]], float(co_vals[i])) for i in order]
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "comissingness"])

def _batch_missing_test(df, groups, group_a, group_b, *, alpha=0.05) -> pd.DataFrame:
    """Per-feature Fisher's exact test of missingness between two groups."""
    from scipy import stats

    mask_a = groups == group_a
    mask_b = groups == group_b
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    cols = ["feature", "log2_OR", "p_value", "q_value", "significant", "enriched_in"]
    if n_a < 2 or n_b < 2:
        return pd.DataFrame(columns=cols)
    M = df.isna().values
    rows = []
    for i, feature in enumerate(df.index):
        miss_a = int(M[i, mask_a].sum())
        miss_b = int(M[i, mask_b].sum())
        pres_a = n_a - miss_a
        pres_b = n_b - miss_b
        if (miss_a + miss_b) < 2:
            continue
        try:
            res = stats.fisher_exact([[miss_a, pres_a], [miss_b, pres_b]])
            p = res.pvalue if hasattr(res, "pvalue") else res[1]
        except Exception:
            continue
        a, b, c, d = miss_a + 0.5, pres_a + 0.5, miss_b + 0.5, pres_b + 0.5
        log2_OR = float(np.log2((a / b) / (c / d)))
        rows.append((feature, log2_OR, float(p)))
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=["feature", "log2_OR", "p_value"])
    m = len(out)
    ranks = out["p_value"].rank(method="first")
    out["q_value"] = (out["p_value"] * m / ranks).clip(upper=1.0)
    out["significant"] = out["q_value"] < alpha
    out["enriched_in"] = np.where(out["log2_OR"] > 0, str(group_a), str(group_b))
    return out[cols]

def _runorder_trend(sample_missing_rate, run_order):
    from scipy import stats

    x = np.asarray(run_order, dtype=float)
    y = sample_missing_rate.values.astype(float)
    if len(x) < 3 or np.std(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    try:
        r, p = stats.pearsonr(x, y)
    except Exception:
        r, p = 0.0, 1.0
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r),
        "p": float(p),
    }

def _comissing_matrix(df: pd.DataFrame, *, top_n: int = 50) -> pd.DataFrame:
    """Full pairwise co-missingness matrix for the top_n most-missing features.

    Cell (i, j) = fraction of samples where both features i and j are missing.
    Diagonal = per-feature missingness rate.
    """
    M = df.isna().astype(np.int8).values
    if M.shape[0] == 0 or M.shape[1] == 0:
        return pd.DataFrame()
    miss_per_feature = M.sum(axis=1)
    if M.shape[0] > top_n:
        top_idx = np.argsort(miss_per_feature)[::-1][:top_n]
    else:
        top_idx = np.arange(M.shape[0])
    Mf = M[top_idx]
    names = df.index[top_idx].tolist()
    n_samples = M.shape[1]
    co = (Mf @ Mf.T) / n_samples
    return pd.DataFrame(co, index=names, columns=names)
