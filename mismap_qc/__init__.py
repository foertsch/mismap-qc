"""mismap-qc: missing-data validation for proteomics and RNA-Seq.

Public API (re-exported here for backwards compatibility):
    qc, assert_qc            -- top-level validation entry points
    MismapReport, RuleResult -- structured results
    MismapQCFailure          -- exception raised by assert_qc
    MismapQCWarning          -- warning category for warning-severity rules
    missing_matrix, missing_matrix_html, missing_abundance_density,
    completeness_bars, detection_waterfall, missing_runorder,
    missing_mechanism, comissing_heatmap -- plot functions
    from_anndata             -- AnnData reader
    estimate_lod             -- per-feature LOD

The package is split into focused submodules. Public names are re-exported
here so ``from mismap_qc import qc`` keeps working. Internal helpers are also
re-exported (with a leading underscore) for tests that historically imported
them.
"""
from __future__ import annotations

__version__ = "0.2.0"

# --- public API ---
from ._core import FEATURE_TYPES
from .io import from_anndata
from .lod import estimate_lod
from .plots import (
    comissing_heatmap,
    completeness_bars,
    detection_waterfall,
    missing_abundance_density,
    missing_matrix,
    missing_matrix_html,
    missing_mechanism,
    missing_runorder,
    rna_missing_matrix,  # legacy alias for missing_matrix
)
from .validation import (
    MismapQCFailure,
    MismapQCWarning,
    MismapReport,
    RuleResult,
    assert_qc,
    qc,
)

# --- internal helpers re-exported for tests / advanced users ---
# noqa flags: these are deliberately re-exported so existing imports keep working.
from ._core import (
    _PALETTES,  # noqa: F401
    _assign_colors,  # noqa: F401
    _clean_ax,  # noqa: F401
    _get_feature_labels,  # noqa: F401
    _resolve_color_overrides,  # noqa: F401
)
from .plots import (
    _RETURN_DATA_SCHEMAS,  # noqa: F401
    _data_comissing_heatmap,  # noqa: F401
    _data_completeness_bars,  # noqa: F401
    _data_detection_waterfall,  # noqa: F401
    _data_missing_matrix,  # noqa: F401
    _data_missing_runorder,  # noqa: F401
    _split_matrix,  # noqa: F401
)
from .stats import (
    _batch_missing_test,  # noqa: F401
    _classify_mechanism,  # noqa: F401
    _comissing_matrix,  # noqa: F401
    _compute_sample_outliers,  # noqa: F401
    _resolve_group_labels,  # noqa: F401
    _runorder_trend,  # noqa: F401
    _top_codropouts,  # noqa: F401
)
from .validation import (
    _RULE_DEFAULT_SEVERITY,  # noqa: F401
    _RULE_EVALUATORS,  # noqa: F401
    _SkipRule,  # noqa: F401
    _evaluate_thresholds,  # noqa: F401
)

__all__ = [
    "qc",
    "assert_qc",
    "MismapReport",
    "RuleResult",
    "MismapQCFailure",
    "MismapQCWarning",
    "missing_matrix",
    "missing_matrix_html",
    "missing_abundance_density",
    "completeness_bars",
    "detection_waterfall",
    "missing_runorder",
    "missing_mechanism",
    "comissing_heatmap",
    "from_anndata",
    "estimate_lod",
    "FEATURE_TYPES",
    "rna_missing_matrix",
]
