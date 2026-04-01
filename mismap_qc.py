"""
mismap-qc: missing-data matrix for proteomics and RNA-Seq QC.

Usage:
    from mismap_qc import missing_matrix

    # df: features (rows) x samples (MultiIndex columns), NaN = missing
    fig = missing_matrix(df, title="Detection Matrix", feature_type="PROT")

    # For genes:
    fig = missing_matrix(df, title="Detection Matrix", feature_type="GENE")

    # Interactive HTML version
    missing_matrix_html(df, title="Detection Matrix", save="output.html")

Feature types: "PROT" (proteins), "GENE" (genes), "PEPTIDE" (peptides)
"""
from __future__ import annotations

__version__ = "0.1.0"

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.cluster import hierarchy


_PALETTES = [
    ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
     "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"],
    ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
     "#E6AB02", "#A6761D", "#666666", "#F781BF", "#A65628"],
    ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
     "#FFFF33", "#A65628", "#F781BF", "#999999", "#66C2A5"],
]

# Feature type labels for axis labels and annotations
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
    rgb_lut = np.array([mpl.colors.to_rgb(cmap[l]) for l in unique])
    idx_map = {u: i for i, u in enumerate(unique)}
    indices = np.array([idx_map[l] for l in labels])
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



# ---------------------------------------------------------------------------
# Main static plot
# ---------------------------------------------------------------------------
def missing_matrix(
    df: pd.DataFrame,
    *,
    title: str = "",
    subtitle: str = "",
    feature_type: str = "PROT",
    annotation_levels: list[int] | None = None,
    annotation_colors: dict[int | str, dict[str, str]] | None = None,
    label_level: int = -1,
    sort_features: str | None = "descending",
    cluster_samples: bool = True,
    cluster_method: str = "average",
    show_dendrogram: bool = True,
    color_present: str | tuple = "#2d2d2d",
    color_missing: str | tuple = "#f0f0f0",
    invert: bool = False,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 10,
    fontsize_legend: int | None = None,
    fontsize_rows: int | None = None,
    fontsize_cols: int | None = None,
    fontsize_annotations: int | None = None,
    completeness: str = "below",
    completeness_threshold: float | None = None,
    legend_loc: str = "upper right",
    group_summary: int | str | None = None,
    split_by: int | str | None = None,
    save: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Pretty missing-data matrix with multi-level sample annotations.

    Parameters
    ----------
    df : DataFrame
        Features (rows) x Samples (columns). Use MultiIndex columns for
        annotation strips; level names become strip labels automatically.
        NaN = missing / not detected.
    title : str
        Figure title (empty string = no title).
    subtitle : str
        Secondary line below title for dataset metadata.
    feature_type : str
        Type of features: "PROT", "GENE", or "PEPTIDE". Used for labels.
    annotation_levels : list[int] | None
        Column-index levels to show as colour bars. Default: all levels
        except the innermost (used for tick labels).
    annotation_colors : dict | None
        Custom colours for annotation levels. Keys are level indices (int)
        or level names (str). Values are dicts mapping factor levels to
        hex colours.
    label_level : int
        Column level for x-axis tick labels (-1 = innermost).
    sort_features : "ascending" | "descending" | None
        Sort features by completeness. Default "descending".
    cluster_samples : bool
        Cluster samples by nullity pattern (default True).
    cluster_method : str
        scipy linkage method (default "average").
    show_dendrogram : bool
        Draw dendrogram above annotations (default True).
    color_present, color_missing : colour spec
        Colours for detected vs missing cells.
    invert : bool
        Swap present and missing colours. When ``True``, missing cells are dark
        and present cells are light — the inverse of the default.
    figsize : tuple | None
        Figure size; auto-calculated if None.
    fontsize : int
        Base font size used as fallback (default 10).
    fontsize_legend : int | None
        Font size for legend entries.
    fontsize_rows : int | None
        Font size for row (feature) labels.
    fontsize_cols : int | None
        Font size for column (sample) labels.
    fontsize_annotations : int | None
        Font size for annotation strip labels.
    completeness : "below" | "side"
        Where to place the completeness sparkline.
    completeness_threshold : float | None
        Draw a threshold line on the sparkline at this value (0-1).
        E.g. 0.5 draws a line at 50% completeness.
    legend_loc : str
        Corner for the annotation legends: "upper right", "upper left",
        "lower right", "lower left" (default "upper right").
    group_summary : int | str | None
        Column level (int index or str name) to group by for a per-group
        completeness summary printed to the console. Only works when there
        is more than one factor level. Default None (disabled).
    split_by : int | str | None
        Split the matrix into side-by-side panels by this column level
        (int index or str name). Each factor value gets its own panel.
    save : str | None
        Save figure to this path if set.
    dpi : int
        Save resolution (default 150).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if invert:
        color_present, color_missing = color_missing, color_present

    # -- handle split_by by delegating to sub-calls -------------------------
    if split_by is not None:
        return _split_matrix(
            df, split_by=split_by, title=title, subtitle=subtitle,
            feature_type=feature_type,
            annotation_levels=annotation_levels,
            annotation_colors=annotation_colors,
            label_level=label_level, sort_features=sort_features,
            cluster_samples=cluster_samples, cluster_method=cluster_method,
            show_dendrogram=show_dendrogram,
            color_present=color_present, color_missing=color_missing,
            figsize=figsize, fontsize=fontsize,
            fontsize_legend=fontsize_legend, fontsize_rows=fontsize_rows,
            fontsize_cols=fontsize_cols, fontsize_annotations=fontsize_annotations,
            completeness=completeness,
            completeness_threshold=completeness_threshold,
            legend_loc=legend_loc, group_summary=group_summary,
            save=save, dpi=dpi,
        )

    # -- resolve font sizes -------------------------------------------------
    fs_legend = fontsize_legend if fontsize_legend is not None else fontsize - 2
    fs_cols = fontsize_cols if fontsize_cols is not None else max(4, fontsize - 3)
    fs_ann = fontsize_annotations if fontsize_annotations is not None else fontsize - 1

    # -- resolve multi-index ------------------------------------------------
    has_mi = isinstance(df.columns, pd.MultiIndex)
    n_levels = df.columns.nlevels if has_mi else 1

    if annotation_levels is None:
        annotation_levels = list(range(n_levels - 1)) if has_mi and n_levels > 1 else []

    level_names = []
    if has_mi:
        for lv in annotation_levels:
            name = df.columns.names[lv]
            level_names.append(name if name is not None else f"Level {lv}")
    n_ann = len(annotation_levels)

    _color_overrides = _resolve_color_overrides(annotation_colors, df.columns)

    # -- nullity mask, sort, cluster ----------------------------------------
    z = df.notnull().values.astype(np.int8)
    n_genes, n_samples = z.shape

    if sort_features == "ascending":
        row_order = np.argsort(z.sum(axis=1))
    elif sort_features == "descending":
        row_order = np.argsort(z.sum(axis=1))[::-1]
    else:
        row_order = np.arange(n_genes)

    col_linkage = None

    if cluster_samples and n_samples > 1:
        col_linkage = hierarchy.linkage(z.T, method=cluster_method)
        col_order = hierarchy.leaves_list(col_linkage)
    else:
        col_order = np.arange(n_samples)

    z = z[np.ix_(row_order, col_order)]
    df = df.iloc[row_order, col_order]

    # -- resolve row font size ----------------------------------------------
    if fontsize_rows is not None:
        fs_rows = fontsize_rows
    elif n_genes <= 120:
        fs_rows = max(3, min(fontsize - 2, int(300 / n_genes)))
    else:
        fs_rows = fontsize - 2

    # -- figure layout ------------------------------------------------------
    show_spark = completeness in ("below", "side")
    spark_below = completeness == "below"
    spark_side = completeness == "side"
    show_dend = show_dendrogram and col_linkage is not None

    if figsize is None:
        w = max(10, n_samples * 0.35 + 4)
        if spark_side:
            w += 2
        h = max(6, n_genes * 0.12 + 2 + n_ann * 0.4)
        figsize = (w, h)

    parts: list[tuple[str, float]] = []
    if show_dend:
        parts.append(("dend", 2.0))
    for lv in annotation_levels:
        parts.append((f"ann_{lv}", 0.4))
    parts.append(("matrix", max(6, n_genes * 0.08)))
    if show_spark and spark_below:
        parts.append(("spark", 1.2))

    if "right" in legend_loc:
        gs_left, gs_right = 0.15, 0.82
    else:
        gs_left, gs_right = 0.22, 0.95

    fig = plt.figure(figsize=figsize, facecolor="white")

    if spark_side:
        outer = gridspec.GridSpec(
            1, 2, width_ratios=[15, 1], wspace=0.08,
            left=gs_left, right=gs_right, top=0.92, bottom=0.06,
        )
        gs = gridspec.GridSpecFromSubplotSpec(
            len(parts), 1,
            height_ratios=[p[1] for p in parts],
            hspace=0.02, subplot_spec=outer[0],
        )
        gs_spark = outer[1]
    else:
        gs = gridspec.GridSpec(
            len(parts), 1,
            height_ratios=[p[1] for p in parts],
            hspace=0.02, left=gs_left, right=gs_right, top=0.92, bottom=0.06,
        )

    axes = {name: fig.add_subplot(gs[i]) for i, (name, _) in enumerate(parts)}

    # -- title + subtitle ---------------------------------------------------
    if title:
        y_title = 0.97
        fig.suptitle(title, fontsize=fontsize + 4, fontweight="bold", y=y_title)
        if subtitle:
            fig.text(0.5, y_title - 0.03, subtitle, ha="center", va="top",
                     fontsize=fontsize, color="#666666", style="italic")

    # -- dendrogram ---------------------------------------------------------
    if show_dend:
        ax = axes["dend"]
        hierarchy.dendrogram(
            col_linkage, orientation="top", no_labels=True,
            link_color_func=lambda _: "#555555",
            above_threshold_color="#555555", ax=ax,
        )
        ax.set_xlim(-0.5, n_samples * 10 - 0.5)
        _clean_ax(ax)
        ax.set_ylabel("Distance", fontsize=fs_ann, labelpad=8)
        ax.tick_params(axis="y", labelsize=fs_legend)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color("#cccccc")

    # -- annotation strips --------------------------------------------------
    legend_handles: list[tuple[str, list[mpatches.Patch]]] = []

    for idx, lv in enumerate(annotation_levels):
        ax = axes[f"ann_{lv}"]
        labels = (np.array(df.columns.get_level_values(lv)) if has_mi
                  else np.array(df.columns.astype(str)))

        rgb_row, cmap = _assign_colors(labels, idx, _color_overrides.get(lv))
        ax.imshow(
            rgb_row.reshape(1, -1, 3), aspect="auto", interpolation="none",
            extent=(-0.5, n_samples - 0.5, 0, 1),
        )
        ax.set_xlim(-0.5, n_samples - 0.5)
        _clean_ax(ax)
        lname = level_names[idx] if idx < len(level_names) else ""
        ax.set_ylabel(lname, fontsize=fs_ann, rotation=0,
                      ha="right", va="center", labelpad=10)

        patches = [mpatches.Patch(facecolor=c, edgecolor="#888", linewidth=0.5,
                                  label=str(lab)) for lab, c in cmap.items()]
        legend_handles.append((lname, patches))

    # -- nullity matrix -----------------------------------------------------
    ax_mat = axes["matrix"]
    c_p = np.array(mpl.colors.to_rgb(color_present), dtype=np.float32)
    c_m = np.array(mpl.colors.to_rgb(color_missing), dtype=np.float32)
    mask3d = z[:, :, np.newaxis].astype(np.float32)
    rgb_mat = mask3d * c_p + (1.0 - mask3d) * c_m

    ax_mat.imshow(
        rgb_mat, aspect="auto", interpolation="none",
        extent=(-0.5, n_samples - 0.5, n_genes - 0.5, -0.5),
    )

    if n_samples > 1:
        ax_mat.vlines(
            np.arange(0.5, n_samples - 0.5), -0.5, n_genes - 0.5,
            colors="white", linewidths=0.3,
        )

    # X tick labels
    sample_labels = (df.columns.get_level_values(label_level) if has_mi
                     else df.columns.astype(str))
    ax_mat.set_xticks(range(n_samples))
    if n_samples <= 80:
        ax_mat.set_xticklabels(sample_labels, rotation=90,
                               fontsize=fs_cols, ha="center")
        ax_mat.xaxis.tick_bottom()
    else:
        ax_mat.set_xticklabels([])

    # Y tick labels
    if n_genes <= 120:
        ax_mat.set_yticks(range(n_genes))
        ax_mat.set_yticklabels(df.index, fontsize=fs_rows)
    else:
        ax_mat.set_yticks([0, n_genes - 1])
        ax_mat.set_yticklabels([1, n_genes], fontsize=fs_rows)

    ax_mat.tick_params(axis="both", length=0)
    for sp in ax_mat.spines.values():
        sp.set_visible(False)

    # -- completeness sparkline ---------------------------------------------
    if show_spark:
        comp = z.sum(axis=0) / n_genes

        if spark_below:
            ax_sp = axes["spark"]
            xs = np.arange(n_samples)
            ax_sp.fill_between(xs, comp, alpha=0.25, color=color_present)
            ax_sp.plot(xs, comp, color=color_present, linewidth=1.2)
            ax_sp.set_xlim(-0.5, n_samples - 0.5)
            ax_sp.set_ylim(0, 1.05)
            ax_sp.set_ylabel("Completeness", fontsize=fs_ann,
                             rotation=0, ha="right", va="center", labelpad=10)
            ax_sp.set_xlabel("Samples", fontsize=fs_ann)
            ax_sp.tick_params(axis="y", labelsize=fs_legend)
            ax_sp.tick_params(axis="x", labelbottom=False, length=0)
            ax_sp.spines["top"].set_visible(False)
            ax_sp.spines["right"].set_visible(False)
            ax_sp.spines["bottom"].set_visible(False)
            ax_sp.grid(axis="y", color="#eee", linewidth=0.5)
            if completeness_threshold is not None:
                ax_sp.axhline(completeness_threshold, color="#CC4444",
                              linestyle="--", linewidth=1.0, alpha=0.8)

        elif spark_side:
            ax_sp = fig.add_subplot(gs_spark)
            gene_comp = z.sum(axis=1) / n_samples
            ys = np.arange(n_genes)
            ax_sp.fill_betweenx(ys, gene_comp, alpha=0.25, color=color_present)
            ax_sp.plot(gene_comp, ys, color=color_present, linewidth=1.2)
            ax_sp.set_ylim(n_genes - 0.5, -0.5)
            ax_sp.set_xlim(0, 1.05)
            ax_sp.set_xlabel("Completeness", fontsize=fs_ann)
            ax_sp.tick_params(axis="x", labelsize=fs_legend)
            ax_sp.tick_params(axis="y", labelleft=False, length=0)
            ax_sp.spines["top"].set_visible(False)
            ax_sp.spines["right"].set_visible(False)
            ax_sp.spines["left"].set_visible(False)
            ax_sp.grid(axis="x", color="#eee", linewidth=0.5)
            if completeness_threshold is not None:
                ax_sp.axvline(completeness_threshold, color="#CC4444",
                              linestyle="--", linewidth=1.0, alpha=0.8)

    # -- legends (tightly stacked in chosen corner) -------------------------
    legend_handles.append(("", [
        mpatches.Patch(facecolor=color_present, edgecolor="#888",
                       linewidth=0.5, label="Detected"),
        mpatches.Patch(facecolor=color_missing, edgecolor="#888",
                       linewidth=0.5, label="Missing"),
    ]))

    _LOC_MAP = {
        "upper right":  (0.83, "top"),
        "upper left":   (0.01, "top"),
        "lower right":  (0.83, "bottom"),
        "lower left":   (0.01, "bottom"),
    }
    loc_x, loc_valign = _LOC_MAP.get(legend_loc, _LOC_MAP["upper right"])

    renderer = fig.canvas.get_renderer()
    drawn_legs = []
    for lname, patches in legend_handles:
        leg = fig.legend(
            handles=patches,
            title=lname if lname else None,
            title_fontsize=fs_legend + 1, fontsize=fs_legend,
            loc="center left" if "right" in legend_loc else "center right",
            bbox_to_anchor=(loc_x, 0.5),
            frameon=True, fancybox=True, edgecolor="#ccc",
            borderpad=0.4, labelspacing=0.3, handletextpad=0.4,
        )
        leg._legend_box.align = "left"
        fig.add_artist(leg)
        drawn_legs.append(leg)

    fig.canvas.draw_idle()
    leg_heights = []
    for leg in drawn_legs:
        bb = leg.get_window_extent(renderer)
        bb_fig = bb.transformed(fig.transFigure.inverted())
        leg_heights.append(bb_fig.height)

    gap = 0.01
    if loc_valign == "top":
        y_cursor = 0.95
        for leg, lh in zip(drawn_legs, leg_heights):
            leg.set_bbox_to_anchor((loc_x, y_cursor - lh / 2),
                                   transform=fig.transFigure)
            y_cursor -= lh + gap
    else:
        y_cursor = 0.05
        for leg, lh in reversed(list(zip(drawn_legs, leg_heights))):
            leg.set_bbox_to_anchor((loc_x, y_cursor + lh / 2),
                                   transform=fig.transFigure)
            y_cursor += lh + gap

    # -- per-group completeness summary (printed to console) -----------------
    if group_summary is not None and has_mi:
        # Resolve the group level
        if isinstance(group_summary, str):
            grp_lv = list(df.columns.names).index(group_summary)
        else:
            grp_lv = group_summary
        labels = np.array(df.columns.get_level_values(grp_lv))
        groups = list(dict.fromkeys(labels))
        if len(groups) > 1:
            grp_name = df.columns.names[grp_lv] or f"Level {grp_lv}"
            print(f"\nGroup Completeness ({grp_name})")
            print("-" * 32)
            for group in groups:
                mask = labels == group
                grp_comp = z[:, mask].sum() / (n_genes * mask.sum()) if mask.any() else 0
                n_samp = mask.sum()
                print(f"  {str(group):14s} {grp_comp:>5.0%}  (n={n_samp})")
            print()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# ---------------------------------------------------------------------------
# Split-by-factor: side-by-side panels
# ---------------------------------------------------------------------------
def _split_matrix(df, *, split_by, title, subtitle, save, dpi, figsize, **kwargs):
    """Render one panel per factor level, arranged side by side."""
    has_mi = isinstance(df.columns, pd.MultiIndex)
    if not has_mi:
        raise ValueError("split_by requires MultiIndex columns")

    # Resolve split_by to int level
    if isinstance(split_by, str):
        split_lv = list(df.columns.names).index(split_by)
    else:
        split_lv = split_by

    split_name = df.columns.names[split_lv] or f"Level {split_lv}"
    groups = list(dict.fromkeys(df.columns.get_level_values(split_lv)))
    n_panels = len(groups)

    # Filter split level out of annotation_levels
    ann_levels = kwargs.get("annotation_levels")
    if ann_levels is None:
        n_levels = df.columns.nlevels
        ann_levels = [i for i in range(n_levels - 1) if i != split_lv]
    else:
        ann_levels = [i for i in ann_levels if i != split_lv]
    kwargs["annotation_levels"] = ann_levels

    # Figure: side by side
    if figsize is None:
        per_panel_w = max(6, df.shape[1] / n_panels * 0.35 + 3)
        h = max(6, df.shape[0] * 0.12 + 3)
        figsize = (per_panel_w * n_panels + 1, h)

    fig, panel_axes = plt.subplots(1, n_panels, figsize=figsize, facecolor="white")
    if n_panels == 1:
        panel_axes = [panel_axes]

    y_title = 0.97
    if title:
        fig.suptitle(title, fontsize=kwargs.get("fontsize", 10) + 4,
                     fontweight="bold", y=y_title)
    if subtitle:
        fig.text(0.5, y_title - 0.03, subtitle, ha="center", va="top",
                 fontsize=kwargs.get("fontsize", 10), color="#666666",
                 style="italic")

    plt.close(fig)  # we'll build sub-figures instead

    # Build individual panels as separate figures, then composite
    panel_figs = []
    for group in groups:
        mask = df.columns.get_level_values(split_lv) == group
        df_sub = df.loc[:, mask]
        sub_fig = missing_matrix(
            df_sub,
            title=f"{split_name}: {group}",
            subtitle="",
            split_by=None,
            save=None,
            **kwargs,
        )
        panel_figs.append(sub_fig)

    # Composite: save each panel then arrange
    # For simplicity, use a fresh figure with subplots showing the panel images
    import io
    images = []
    for sf in panel_figs:
        buf = io.BytesIO()
        sf.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(sf)
        buf.seek(0)
        img = plt.imread(buf)
        images.append(img)
        buf.close()

    max_h = max(im.shape[0] for im in images)
    total_w = sum(im.shape[1] for im in images)

    comp_fig, comp_axes = plt.subplots(
        1, n_panels,
        figsize=(total_w / dpi, max_h / dpi),
        facecolor="white",
    )
    if n_panels == 1:
        comp_axes = [comp_axes]

    for ax, img in zip(comp_axes, images):
        ax.imshow(img)
        _clean_ax(ax)

    comp_fig.subplots_adjust(wspace=0.02, left=0, right=1, top=1, bottom=0)

    if title:
        comp_fig.suptitle(title, fontsize=kwargs.get("fontsize", 10) + 4,
                          fontweight="bold", y=1.02)

    if save:
        comp_fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")

    return comp_fig


# ---------------------------------------------------------------------------
# Interactive HTML export (plotly)
# ---------------------------------------------------------------------------
def missing_matrix_html(
    df: pd.DataFrame,
    *,
    title: str = "Missing Data Matrix",
    subtitle: str = "",
    feature_type: str = "PROT",
    annotation_levels: list[int] | None = None,
    annotation_colors: dict[int | str, dict[str, str]] | None = None,
    label_level: int = -1,
    sort_features: str | None = "descending",
    cluster_samples: bool = True,
    cluster_method: str = "average",
    color_present: str = "#2d2d2d",
    color_missing: str = "#f0f0f0",
    invert: bool = False,
    completeness: str = "below",
    completeness_threshold: float | None = None,
    width: int | None = None,
    height: int | None = None,
    save: str | None = None,
) -> str:
    """
    Interactive HTML missing-data matrix using plotly.

    Parameters
    ----------
    feature_type : str
        Type of features: "PROT", "GENE", or "PEPTIDE". Used for hover labels.

    Returns the HTML string. If save is set, also writes to file.
    """
    if invert:
        color_present, color_missing = color_missing, color_present

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("pip install plotly  -- required for HTML export")

    fl = _get_feature_labels(feature_type)
    has_mi = isinstance(df.columns, pd.MultiIndex)
    n_levels = df.columns.nlevels if has_mi else 1

    if annotation_levels is None:
        annotation_levels = list(range(n_levels - 1)) if has_mi and n_levels > 1 else []

    level_names = []
    if has_mi:
        for lv in annotation_levels:
            name = df.columns.names[lv]
            level_names.append(name if name is not None else f"Level {lv}")

    _color_overrides = _resolve_color_overrides(annotation_colors, df.columns)

    # -- nullity mask, sort, cluster ----------------------------------------
    z = df.notnull().values.astype(np.int8)
    n_genes, n_samples = z.shape

    if sort_features == "ascending":
        row_order = np.argsort(z.sum(axis=1))
    elif sort_features == "descending":
        row_order = np.argsort(z.sum(axis=1))[::-1]
    else:
        row_order = np.arange(n_genes)

    if cluster_samples and n_samples > 1:
        col_linkage = hierarchy.linkage(z.T, method=cluster_method)
        col_order = hierarchy.leaves_list(col_linkage)
    else:
        col_order = np.arange(n_samples)

    z = z[np.ix_(row_order, col_order)]
    df = df.iloc[row_order, col_order]

    # -- sample labels & hover text -----------------------------------------
    if has_mi:
        sample_labels = [str(x) for x in df.columns.get_level_values(label_level)]
        hover_parts = []
        for lv in range(n_levels):
            lname = df.columns.names[lv] or f"Level {lv}"
            vals = df.columns.get_level_values(lv)
            hover_parts.append((lname, vals))
    else:
        sample_labels = [str(c) for c in df.columns]
        hover_parts = []

    feature_labels = [str(g) for g in df.index]

    # Build hover text matrix
    hover_text = []
    for i in range(n_genes):
        row = []
        for j in range(n_samples):
            parts = [f"<b>{fl['cap_singular']}:</b> {feature_labels[i]}"]
            parts.append(f"<b>Sample:</b> {sample_labels[j]}")
            for lname, vals in hover_parts:
                parts.append(f"<b>{lname}:</b> {vals[j]}")
            status = "Detected" if z[i, j] else "Missing"
            parts.append(f"<b>Status:</b> {status}")
            row.append("<br>".join(parts))
        hover_text.append(row)

    # -- build plotly figure ------------------------------------------------
    n_ann = len(annotation_levels)
    show_spark = completeness in ("below", "side")
    spark_below = completeness == "below"

    # Row layout: annotations + matrix + optional completeness bar
    subplot_rows = n_ann + 1 + (1 if show_spark and spark_below else 0)
    ann_height = 0.02
    spark_height = 0.12 if show_spark and spark_below else 0
    matrix_height = 1.0 - ann_height * n_ann - spark_height
    row_heights = [ann_height] * n_ann + [matrix_height]
    if show_spark and spark_below:
        row_heights.append(spark_height)

    fig = make_subplots(
        rows=subplot_rows, cols=1,
        row_heights=row_heights,
        vertical_spacing=0.008,
        shared_xaxes=True,
    )

    # -- annotation strips (discrete colored cells) -------------------------
    for idx, lv in enumerate(annotation_levels):
        labels = np.array(df.columns.get_level_values(lv)) if has_mi else np.array(sample_labels)
        _, cmap = _assign_colors(labels, idx, _color_overrides.get(lv))

        # Use individual colored rectangles for crisp annotation strips
        unique_labels = list(dict.fromkeys(labels))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        z_ann = [[label_to_int[l] for l in labels]]

        # Build discrete colorscale
        n_unique = len(unique_labels)
        discrete_cs = []
        for i, ul in enumerate(unique_labels):
            lo = i / n_unique
            hi = (i + 1) / n_unique
            discrete_cs.append([lo, cmap[ul]])
            discrete_cs.append([hi, cmap[ul]])

        fig.add_trace(go.Heatmap(
            z=z_ann,
            x=sample_labels,
            colorscale=discrete_cs,
            showscale=False,
            xgap=1,
            hovertext=[[f"<b>{level_names[idx]}:</b> {l}" for l in labels]],
            hoverinfo="text",
        ), row=idx + 1, col=1)

        fig.update_yaxes(
            title_text=level_names[idx] if idx < len(level_names) else "",
            title_font=dict(size=11),
            showticklabels=False, row=idx + 1, col=1,
        )

    # -- main heatmap -------------------------------------------------------
    matrix_row = n_ann + 1
    colorscale = [[0, color_missing], [1, color_present]]
    fig.add_trace(go.Heatmap(
        z=z,
        x=sample_labels,
        y=feature_labels,
        colorscale=colorscale,
        showscale=False,
        hovertext=hover_text,
        hoverinfo="text",
        xgap=1,
        ygap=0,
    ), row=matrix_row, col=1)

    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=max(6, min(10, int(400 / n_genes)))),
        row=matrix_row, col=1,
    )

    # -- completeness sparkline (filled area, matching static PNG style) -----
    if show_spark and spark_below:
        spark_row = matrix_row + 1
        comp = z.sum(axis=0) / n_genes
        xs = list(range(n_samples))

        # Filled area under the line (numeric x for smooth curve)
        fig.add_trace(go.Scatter(
            x=xs,
            y=comp,
            fill="tozeroy",
            fillcolor="rgba(45,45,45,0.25)",
            line=dict(color=color_present, width=1.2),
            hovertext=[f"<b>{sample_labels[i]}</b>: {comp[i]:.1%}" for i in range(n_samples)],
            hoverinfo="text",
            showlegend=False,
        ), row=spark_row, col=1)

        if completeness_threshold is not None:
            fig.add_hline(
                y=completeness_threshold, row=spark_row, col=1,
                line=dict(color="#CC4444", width=1.5, dash="dash"),
            )

        fig.update_yaxes(
            title_text="Completeness", title_font=dict(size=11),
            range=[0, 1.05], tickformat=".0%",
            row=spark_row, col=1,
        )
        fig.update_xaxes(
            range=[-0.5, n_samples - 0.5],
            showticklabels=False,
            title_text="Samples", title_font=dict(size=11),
            row=spark_row, col=1,
        )

    # -- annotation legends (shapes in margin) ------------------------------
    # Build legend annotations for each level
    legend_annotations = []
    y_legend = 0.98
    for idx, lv in enumerate(annotation_levels):
        labels = np.array(df.columns.get_level_values(lv)) if has_mi else np.array(sample_labels)
        _, cmap = _assign_colors(labels, idx, _color_overrides.get(lv))
        lname = level_names[idx] if idx < len(level_names) else ""

        # Title
        legend_annotations.append(dict(
            x=1.02, y=y_legend, xref="paper", yref="paper",
            text=f"<b>{lname}</b>", showarrow=False,
            font=dict(size=11), xanchor="left",
        ))
        y_legend -= 0.025

        for label, color in cmap.items():
            legend_annotations.append(dict(
                x=1.02, y=y_legend, xref="paper", yref="paper",
                text=f'<span style="color:{color};">\u25a0</span> {label}',
                showarrow=False, font=dict(size=10), xanchor="left",
            ))
            y_legend -= 0.022
        y_legend -= 0.015

    # Detected / Missing legend
    legend_annotations.append(dict(
        x=1.02, y=y_legend, xref="paper", yref="paper",
        text=f'<span style="color:{color_present};">\u25a0</span> Detected',
        showarrow=False, font=dict(size=10), xanchor="left",
    ))
    y_legend -= 0.022
    legend_annotations.append(dict(
        x=1.02, y=y_legend, xref="paper", yref="paper",
        text=f'<span style="color:{color_missing};">\u25a0</span> Missing',
        showarrow=False, font=dict(size=10), xanchor="left",
    ))

    # -- title + subtitle ---------------------------------------------------
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:13px;color:#666666;'><i>{subtitle}</i></span>"

    # -- layout -------------------------------------------------------------
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16), x=0.5, xanchor="center"),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width or max(800, n_samples * 25 + 250),
        height=height or max(600, n_genes * 8 + 200),
        margin=dict(l=120, r=160, t=80, b=40),
        annotations=legend_annotations,
    )

    # Style all axes
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    if save:
        with open(save, "w") as f:
            f.write(html)
    return html


# ---------------------------------------------------------------------------
# Abundance-by-missingness density plot
# ---------------------------------------------------------------------------
def missing_abundance_density(
    df: pd.DataFrame,
    *,
    groups: pd.Series | np.ndarray | list | None = None,
    max_na_levels: int = 6,
    title: str = "Abundance by Missingness",
    xlabel: str = "Mean Abundance",
    ylabel: str = "Density",
    figsize: tuple[float, float] | None = None,
    fontsize: int = 10,
    palette: list[str] | None = None,
    alpha: float = 0.7,
    linewidth: float = 1.5,
    legend_title: str = "# Missing",
    save: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Density plot of mean abundance stratified by missingness count.

    This diagnostic plot reveals whether missing data follows the MNAR
    (Missing Not At Random) pattern typical of proteomics/mass-spec data,
    where low-abundance features are more likely to be missing due to
    detection limits.

    Parameters
    ----------
    df : DataFrame
        Genes/proteins (rows) x Samples (columns). NaN = missing.
    groups : Series, array, or list, optional
        Group labels for each sample (same length as df.columns).
        If provided, creates faceted subplots, one per group.
    max_na_levels : int
        Maximum number of distinct missingness levels to show (default 6).
        Higher counts are binned into "N+" category.
    title : str
        Figure title.
    xlabel, ylabel : str
        Axis labels.
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    fontsize : int
        Base font size.
    palette : list[str], optional
        Colors for missingness levels. Uses built-in palette if None.
    alpha : float
        Line/fill transparency (default 0.7).
    linewidth : float
        Density line width (default 1.5).
    legend_title : str
        Legend title (default "# Missing").
    save : str, optional
        Save figure to this path.
    dpi : int
        Save resolution (default 150).

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = missing_abundance_density(df)
    >>> fig = missing_abundance_density(df, groups=df.columns.get_level_values("Condition"))
    """
    n_genes, n_samples = df.shape

    # Compute per-gene metrics
    na_counts = df.isna().sum(axis=1).values
    mean_abundance = df.mean(axis=1, skipna=True).values

    # Filter out genes with all missing (no valid mean)
    valid_mask = ~np.isnan(mean_abundance)
    na_counts = na_counts[valid_mask]
    mean_abundance = mean_abundance[valid_mask]
    gene_idx = np.where(valid_mask)[0]

    # Bin high NA counts
    na_labels = na_counts.copy()
    if max_na_levels is not None and na_counts.max() >= max_na_levels:
        na_labels = np.where(
            na_counts >= max_na_levels,
            max_na_levels,
            na_counts
        )

    unique_na = np.sort(np.unique(na_labels))

    # Palette
    if palette is None:
        # Use a sequential colormap for missingness levels
        cmap = plt.cm.viridis_r
        palette = [mpl.colors.to_hex(cmap(i / max(len(unique_na) - 1, 1)))
                   for i in range(len(unique_na))]

    # Handle faceting by groups
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = list(dict.fromkeys(groups))
        n_groups = len(unique_groups)

        if figsize is None:
            figsize = (5 * n_groups, 4)

        fig, axes = plt.subplots(1, n_groups, figsize=figsize,
                                 sharey=True, facecolor="white")
        if n_groups == 1:
            axes = [axes]

        for ax, grp in zip(axes, unique_groups):
            # Get samples in this group
            grp_mask = groups == grp
            grp_samples = np.where(grp_mask)[0]

            # Recompute metrics for this group's samples
            df_grp = df.iloc[:, grp_samples]
            grp_na = df_grp.isna().sum(axis=1).values
            grp_mean = df_grp.mean(axis=1, skipna=True).values

            valid = ~np.isnan(grp_mean)
            grp_na = grp_na[valid]
            grp_mean = grp_mean[valid]

            # Bin
            grp_na_labels = grp_na.copy()
            if max_na_levels is not None and grp_na.max() >= max_na_levels:
                grp_na_labels = np.where(
                    grp_na >= max_na_levels, max_na_levels, grp_na
                )

            grp_unique_na = np.sort(np.unique(grp_na_labels))

            for i, na_val in enumerate(grp_unique_na):
                mask = grp_na_labels == na_val
                if mask.sum() < 2:
                    continue
                values = grp_mean[mask]

                # KDE
                from scipy import stats
                try:
                    kde = stats.gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 200)
                    density = kde(x_range)

                    label = f"{na_val}" if na_val < max_na_levels else f"{max_na_levels}+"
                    color = palette[min(i, len(palette) - 1)]
                    ax.fill_between(x_range, density, alpha=alpha * 0.4, color=color)
                    ax.plot(x_range, density, color=color, linewidth=linewidth,
                            label=label)
                except np.linalg.LinAlgError:
                    # KDE can fail with singular matrix
                    pass

            ax.set_title(str(grp), fontsize=fontsize + 1)
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel(ylabel, fontsize=fontsize)

        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title=legend_title,
                   loc="upper right", bbox_to_anchor=(0.98, 0.95),
                   fontsize=fontsize - 1, title_fontsize=fontsize)

    else:
        # Single panel
        if figsize is None:
            figsize = (7, 5)

        fig, ax = plt.subplots(figsize=figsize, facecolor="white")

        from scipy import stats

        for i, na_val in enumerate(unique_na):
            mask = na_labels == na_val
            if mask.sum() < 2:
                continue
            values = mean_abundance[mask]

            try:
                kde = stats.gaussian_kde(values)
                x_range = np.linspace(values.min(), values.max(), 200)
                density = kde(x_range)

                label = f"{na_val}" if na_val < max_na_levels else f"{max_na_levels}+"
                color = palette[min(i, len(palette) - 1)]
                ax.fill_between(x_range, density, alpha=alpha * 0.4, color=color)
                ax.plot(x_range, density, color=color, linewidth=linewidth,
                        label=label)
            except np.linalg.LinAlgError:
                pass

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title=legend_title, fontsize=fontsize - 1,
                  title_fontsize=fontsize)

    if title:
        fig.suptitle(title, fontsize=fontsize + 2, fontweight="bold", y=1.02)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# Keep old name as alias
rna_missing_matrix = missing_matrix


# ---------------------------------------------------------------------------
# Per-group completeness bar chart
# ---------------------------------------------------------------------------
def completeness_bars(
    df: pd.DataFrame,
    group_level: int | str,
    *,
    threshold: float | None = None,
    color: str | dict | None = None,
    orientation: str = "horizontal",
    title: str = "Per-Group Completeness",
    fontsize: int = 10,
    save: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Horizontal (or vertical) bar chart of per-group detection completeness.

    For each group, shows the mean fraction of features detected across all
    samples in that group. Replaces the console-only ``group_summary`` output
    of ``missing_matrix`` with a publishable figure.

    Parameters
    ----------
    df : DataFrame
        Features (rows) x Samples (columns). NaN = missing / not detected.
        Columns may be a MultiIndex; use ``group_level`` to select the grouping.
    group_level : int or str
        Column level (index or name) to group samples by.
        If ``df`` has flat columns, pass ``0`` or any label — the whole dataset
        is treated as one group (useful for a single-bar sanity check).
    threshold : float or None
        Draw a dashed red line at this completeness value (0–1). E.g. ``0.7``
        marks the 70% completeness threshold.
    color : str, dict, or None
        Single hex colour for all bars, or ``{group_label: hex}`` dict for
        per-group colours. Uses the built-in palette if ``None``.
    orientation : "horizontal" or "vertical"
        Bar orientation. Horizontal (default) is easier to read with long
        group names.
    title : str
        Figure title.
    fontsize : int
        Base font size (default 10).
    save : str or None
        Save figure to this path if set.
    dpi : int
        Save resolution (default 150).

    Returns
    -------
    matplotlib.figure.Figure
    """
    has_mi = isinstance(df.columns, pd.MultiIndex)

    # Resolve groups
    if has_mi:
        if isinstance(group_level, str):
            grp_lv = list(df.columns.names).index(group_level)
        else:
            grp_lv = group_level
        labels = np.array(df.columns.get_level_values(grp_lv))
    else:
        labels = np.array(["All samples"] * len(df.columns))

    groups = list(dict.fromkeys(labels))

    # Compute completeness per group
    completeness = {}
    for grp in groups:
        mask = labels == grp
        completeness[grp] = float(df.loc[:, mask].notna().mean().mean())

    # Sort descending
    groups_sorted = sorted(groups, key=lambda g: completeness[g], reverse=True)
    values = [completeness[g] for g in groups_sorted]

    # Resolve colours
    if isinstance(color, dict):
        colours = [color.get(g, "#4C72B0") for g in groups_sorted]
    elif isinstance(color, str):
        colours = [color] * len(groups_sorted)
    else:
        _, cmap = _assign_colors(np.array(groups_sorted), 0)
        colours = [cmap[g] for g in groups_sorted]

    # Figure
    n_groups = len(groups_sorted)
    if orientation == "horizontal":
        figsize = (7, max(3, n_groups * 0.5 + 1))
    else:
        figsize = (max(4, n_groups * 0.7 + 1), 5)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    if orientation == "horizontal":
        bars = ax.barh(range(n_groups), values, color=colours, edgecolor="white",
                       linewidth=0.5)
        ax.set_yticks(range(n_groups))
        ax.set_yticklabels(groups_sorted, fontsize=fontsize)
        ax.set_xlabel("Completeness", fontsize=fontsize)
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        if threshold is not None:
            ax.axvline(threshold, color="#CC4444", linestyle="--",
                       linewidth=1.2, alpha=0.9,
                       label=f"{threshold:.0%} threshold")
            ax.legend(fontsize=fontsize - 1)
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", ha="left", fontsize=fontsize - 1)
    else:
        bars = ax.bar(range(n_groups), values, color=colours, edgecolor="white",
                      linewidth=0.5)
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(groups_sorted, fontsize=fontsize, rotation=45, ha="right")
        ax.set_ylabel("Completeness", fontsize=fontsize)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        if threshold is not None:
            ax.axhline(threshold, color="#CC4444", linestyle="--",
                       linewidth=1.2, alpha=0.9,
                       label=f"{threshold:.0%} threshold")
            ax.legend(fontsize=fontsize - 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.1%}", va="bottom", ha="center", fontsize=fontsize - 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=10)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# ---------------------------------------------------------------------------
# detection_waterfall — feature detection threshold curve
# ---------------------------------------------------------------------------


def detection_waterfall(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
    group_level: int | str | None = None,
    feature_type: str = "PROT",
    color: str = "#2d2d2d",
    title: str | None = None,
    subtitle: str = "",
    figsize: tuple[float, float] | None = None,
    fontsize: int = 10,
    save: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Waterfall plot showing features ranked by detection rate.

    Features are ranked by their detection rate across samples and plotted as
    a cumulative curve. Threshold lines show how many features survive at
    different filtering cutoffs.

    Parameters
    ----------
    df : pandas.DataFrame
        Features (rows) x samples (columns). NaN = missing/not detected.
    thresholds : list of float or None
        Detection rate thresholds to draw as horizontal lines (0-1 scale).
        Default: [0.5, 0.7, 0.9].
    group_level : int, str, or None
        If set, compute and plot separate curves per group (MultiIndex level).
    feature_type : str
        Type of features: "PROT", "GENE", or "PEPTIDE". Used for axis labels.
    color : str
        Colour for the curve when not grouping. Default "#2d2d2d".
    title : str or None
        Figure title. Auto-generated from feature_type if None.
    subtitle : str
        Italic line below title.
    figsize : tuple or None
        Figure size. Auto-calculated if None.
    fontsize : int
        Base font size (default 10).
    save : str or None
        Save figure to this path if set.
    dpi : int
        Save resolution (default 150).

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = detection_waterfall(df, thresholds=[0.5, 0.7])
    >>> fig = detection_waterfall(df, group_level="Condition", feature_type="GENE")
    """
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.9]

    fl = _get_feature_labels(feature_type)
    if title is None:
        title = f"{fl['cap_singular']} Detection Waterfall"

    has_mi = isinstance(df.columns, pd.MultiIndex)

    if figsize is None:
        figsize = (8, 5)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # If grouping, plot one curve per group
    if group_level is not None and has_mi:
        if isinstance(group_level, str):
            grp_lv = list(df.columns.names).index(group_level)
        else:
            grp_lv = group_level
        labels = np.array(df.columns.get_level_values(grp_lv))
        groups = list(dict.fromkeys(labels))

        _, cmap = _assign_colors(np.array(groups), 0)

        for grp in groups:
            mask = labels == grp
            df_grp = df.loc[:, mask]
            detection_rates = df_grp.notna().mean(axis=1).sort_values(ascending=False)
            x = np.arange(len(detection_rates))
            y = detection_rates.values

            ax.fill_between(x, y, alpha=0.3, color=cmap[grp])
            ax.plot(x, y, linewidth=1.5, color=cmap[grp], label=grp)
    else:
        # Single curve for all samples
        detection_rates = df.notna().mean(axis=1).sort_values(ascending=False)
        x = np.arange(len(detection_rates))
        y = detection_rates.values

        ax.fill_between(x, y, alpha=0.3, color=color)
        ax.plot(x, y, linewidth=1.5, color=color)

    # Draw threshold lines with annotations
    n_features = len(df)
    for thresh in sorted(thresholds, reverse=True):
        ax.axhline(thresh, color="#CC4444", linestyle="--", linewidth=1, alpha=0.8)

        # Count features at or above this threshold
        n_above = int((df.notna().mean(axis=1) >= thresh).sum())
        pct = n_above / n_features * 100

        # Position annotation at right edge
        ax.text(
            n_features * 0.98, thresh + 0.02,
            f"{n_above:,} {fl['plural']} ({pct:.0f}%) at ≥{thresh:.0%}",
            ha="right", va="bottom", fontsize=fontsize - 1, color="#CC4444"
        )

    # Axis formatting
    ax.set_xlim(0, n_features)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"{fl['cap_plural']} (ranked by detection rate)", fontsize=fontsize)
    ax.set_ylabel("Detection rate", fontsize=fontsize)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.tick_params(labelsize=fontsize - 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend if grouping
    if group_level is not None and has_mi:
        ax.legend(fontsize=fontsize - 1, loc="lower left")

    # Title and subtitle
    if title:
        ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=10)
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fontsize - 1, fontstyle="italic", color="#666666"
        )

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig


# ---------------------------------------------------------------------------
# missing_runorder — missingness over run order / time
# ---------------------------------------------------------------------------


def missing_runorder(
    df: pd.DataFrame,
    run_order: list | pd.Series | np.ndarray | None = None,
    group_level: int | str | None = None,
    smooth: bool = True,
    smooth_window: int = 5,
    title: str = "Missingness Over Run Order",
    subtitle: str = "",
    figsize: tuple[float, float] | None = None,
    fontsize: int = 10,
    save: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot per-sample missingness rate against run order.

    Shows how missingness varies across sample acquisition order. Useful for
    detecting instrument drift or batch effects in proteomics experiments.

    Parameters
    ----------
    df : pandas.DataFrame
        Features (rows) x samples (columns). NaN = missing/not detected.
    run_order : array-like or None
        Explicit run order values (one per sample). Uses column index if None.
    group_level : int, str, or None
        If set, colour points by this MultiIndex level (e.g. batch, condition).
    smooth : bool
        Add a rolling mean smoother line. Default True.
    smooth_window : int
        Window size for rolling mean. Default 5.
    title : str
        Figure title.
    subtitle : str
        Italic line below title.
    figsize : tuple or None
        Figure size. Auto-calculated if None.
    fontsize : int
        Base font size (default 10).
    save : str or None
        Save figure to this path if set.
    dpi : int
        Save resolution (default 150).

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = missing_runorder(df)
    >>> fig = missing_runorder(df, group_level="Batch", smooth=True)
    """
    has_mi = isinstance(df.columns, pd.MultiIndex)
    n_samples = len(df.columns)

    # Per-sample missingness rate
    missing_rate = df.isna().mean(axis=0).values

    # Run order
    if run_order is None:
        x = np.arange(n_samples)
        x_label = "Sample index"
    else:
        x = np.array(run_order)
        x_label = "Run order"

    if figsize is None:
        figsize = (10, 5)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # Plot points, optionally coloured by group
    if group_level is not None and has_mi:
        if isinstance(group_level, str):
            grp_lv = list(df.columns.names).index(group_level)
        else:
            grp_lv = group_level
        labels = np.array(df.columns.get_level_values(grp_lv))
        groups = list(dict.fromkeys(labels))

        _, cmap = _assign_colors(np.array(groups), 0)

        for grp in groups:
            mask = labels == grp
            ax.scatter(x[mask], missing_rate[mask], c=cmap[grp], label=grp,
                       s=40, alpha=0.7, edgecolors="white", linewidths=0.5)
    else:
        ax.scatter(x, missing_rate, c="#2d2d2d", s=40, alpha=0.7,
                   edgecolors="white", linewidths=0.5)

    # Smoother line (rolling mean)
    if smooth and n_samples >= smooth_window:
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = missing_rate[sorted_idx]
        y_smooth = pd.Series(y_sorted).rolling(
            window=smooth_window, center=True, min_periods=1
        ).mean().values
        ax.plot(x_sorted, y_smooth, color="#CC4444", linewidth=2, alpha=0.8,
                label=f"Rolling mean (n={smooth_window})")

    # Dataset mean line
    mean_missing = missing_rate.mean()
    ax.axhline(mean_missing, color="#666666", linestyle="--", linewidth=1,
               alpha=0.8, label=f"Mean: {mean_missing:.1%}")

    # Axis formatting
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(0, min(1.05, missing_rate.max() * 1.2 + 0.05))
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel("Missing rate per sample", fontsize=fontsize)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.tick_params(labelsize=fontsize - 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    ax.legend(fontsize=fontsize - 1, loc="upper right")

    # Title and subtitle
    if title:
        ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=10)
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fontsize - 1, fontstyle="italic", color="#666666"
        )

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")

    return fig
