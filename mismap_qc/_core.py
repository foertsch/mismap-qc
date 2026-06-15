"""Core helpers: palettes, feature-type labels, axis layout."""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd


_PALETTES = [
    ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
     "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
     "#E6AB02", "#A6761D", "#666666", "#F781BF", "#A65628"],
    ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
     "#FFFF33", "#A65628", "#F781BF", "#999999", "#66C2A5"],
]

FEATURE_TYPES = {
    "PROT": {"singular": "protein", "plural": "proteins", "cap_singular": "Protein", "cap_plural": "Proteins"},
    "GENE": {"singular": "gene", "plural": "genes", "cap_singular": "Gene", "cap_plural": "Genes"},
    "PEPTIDE": {"singular": "peptide", "plural": "peptides", "cap_singular": "Peptide", "cap_plural": "Peptides"},
}

def _get_feature_labels(feature_type: str) -> dict:
    """Get label strings for a feature type. Returns PROT labels if unknown."""
    return FEATURE_TYPES.get(feature_type.upper(), FEATURE_TYPES["PROT"])

def _assign_colors(
    labels: np.ndarray,
    palette_idx: int,
    overrides: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict]:
    palette = _PALETTES[palette_idx % len(_PALETTES)]
    unique = list(dict.fromkeys(labels))
    cmap = {u: palette[i % len(palette)] for i, u in enumerate(unique)}
    if overrides:
        for k, v in overrides.items():
            if k in cmap:
                cmap[k] = v
    rgb_lut = np.array([mpl.colors.to_rgb(cmap[lab]) for lab in unique])
    idx_map = {u: i for i, u in enumerate(unique)}
    indices = np.array([idx_map[lab] for lab in labels])
    return rgb_lut[indices], cmap

def _clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

def _resolve_color_overrides(annotation_colors, df_columns):
    """Build {int_level: {factor: hex}} lookup from user's annotation_colors."""
    out: dict[int, dict[str, str]] = {}
    if not annotation_colors or not isinstance(df_columns, pd.MultiIndex):
        return out
    for key, val in annotation_colors.items():
        if isinstance(key, int):
            out[key] = val
        elif isinstance(key, str):
            for i, n in enumerate(df_columns.names):
                if n == key:
                    out[i] = val
                    break
    return out
