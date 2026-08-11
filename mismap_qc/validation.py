"""Validation API: MismapReport, qc(), assert_qc(), and the 11-rule registry."""
from __future__ import annotations

import json as _json
import warnings as _warnings
from dataclasses import asdict, dataclass, replace

import numpy as np  # noqa: F401  (used by some rule evaluators via report fields)
import pandas as pd

from .stats import (
    _batch_missing_test,
    _classify_mechanism,
    _compute_sample_outliers,
    _resolve_group_labels,
    _runorder_trend,
    _top_codropouts,
)


class MismapQCWarning(UserWarning):
    """Emitted when a warning-severity QC rule is violated."""

@dataclass(frozen=True)
class RuleResult:
    """A single threshold-rule evaluation."""
    rule: str
    severity: str  # "error" | "warning" | "info"
    passed: bool
    threshold: float
    actual: float
    detail: str = ""

_RULE_DEFAULT_SEVERITY = {
    "min_sample_completeness": "error",
    "min_sample_completeness_per_group": "error",
    "min_features_detected": "error",
    "max_feature_missing_rate": "warning",
    "max_sample_outliers": "error",
    "max_sample_outlier_zscore": "error",
    "max_mnar_fraction": "warning",
    "max_unclassified_fraction": "info",
    "min_group_completeness": "error",
    "max_batch_effect_features": "warning",
    "max_runorder_slope": "warning",
}

@dataclass(frozen=True)
class MismapReport:
    """Immutable snapshot of a QC analysis. Built by qc()."""

    # Inputs (for reproducibility)
    n_features: int
    n_samples: int
    group_level: object = None  # int | str | None; object to keep dataclass simple

    # Per-sample
    sample_missing_rate: pd.Series | None = None
    sample_outliers: pd.DataFrame | None = None

    # Per-feature
    feature_detection_rate: pd.Series | None = None
    feature_mechanism: pd.DataFrame | None = None
    feature_codropouts: pd.DataFrame | None = None

    # Per-group
    group_completeness: pd.Series | None = None

    # Batch / temporal (optional)
    batch_test: pd.DataFrame | None = None
    runorder_trend: dict | None = None

    # Threshold evaluation results
    results: tuple = ()  # tuple[RuleResult, ...]

    # --- computed properties -------------------------------------------------
    @property
    def errors(self):
        return tuple(r for r in self.results if r.severity == "error" and not r.passed)

    @property
    def warnings(self):
        return tuple(r for r in self.results if r.severity == "warning" and not r.passed)

    @property
    def info(self):
        return tuple(r for r in self.results if r.severity == "info" and not r.passed)

    @property
    def passed(self) -> bool:
        return not self.errors

    # --- no-raise threshold evaluation --------------------------------------
    def check(self, thresholds: dict, severity_overrides: dict | None = None):
        """Re-evaluate thresholds against this report. Returns RuleResult tuple, does not raise."""
        return _evaluate_thresholds(self, thresholds, severity_overrides, emit_warnings=False)

    def passes(self, thresholds: dict, severity_overrides: dict | None = None) -> bool:
        """True iff no error-severity rule would fail at these thresholds."""
        results = self.check(thresholds, severity_overrides)
        return not any(r.severity == "error" and not r.passed for r in results)

    # --- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        def _serialize(x):
            if x is None:
                return None
            if isinstance(x, pd.DataFrame):
                return x.to_dict(orient="records")
            if isinstance(x, pd.Series):
                return x.to_dict()
            return x

        return {
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "group_level": self.group_level,
            "sample_missing_rate": _serialize(self.sample_missing_rate),
            "sample_outliers": _serialize(self.sample_outliers),
            "feature_detection_rate": _serialize(self.feature_detection_rate),
            "feature_mechanism": _serialize(self.feature_mechanism),
            "feature_codropouts": _serialize(self.feature_codropouts),
            "group_completeness": _serialize(self.group_completeness),
            "batch_test": _serialize(self.batch_test),
            "runorder_trend": self.runorder_trend,
            "results": [asdict(r) for r in self.results],
            "passed": self.passed,
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        text = _json.dumps(self.to_dict(), default=str, indent=indent)
        if path is not None:
            with open(path, "w") as f:
                f.write(text)
        return text

    def to_html(self, path: str | None = None) -> str:
        verdict = "PASSED" if self.passed else "FAILED"
        rows = [
            f"<h3>MismapReport ({self.n_features} features &times; {self.n_samples} samples)</h3>",
            f"<p><b>Verdict:</b> {verdict}</p>",
        ]
        if self.results:
            rows.append("<h4>Rule results</h4>")
            rows.append("<table border='1' cellpadding='4'>")
            rows.append(
                "<tr><th>Rule</th><th>Severity</th><th>Passed</th>"
                "<th>Threshold</th><th>Actual</th><th>Detail</th></tr>"
            )
            for r in self.results:
                rows.append(
                    f"<tr><td>{r.rule}</td><td>{r.severity}</td>"
                    f"<td>{'&check;' if r.passed else '&cross;'}</td>"
                    f"<td>{r.threshold}</td><td>{r.actual:.4g}</td>"
                    f"<td>{r.detail}</td></tr>"
                )
            rows.append("</table>")
        html = "\n".join(rows)
        if path is not None:
            with open(path, "w") as f:
                f.write(html)
        return html

    # --- string repr --------------------------------------------------------
    def __repr__(self) -> str:
        n_out = 0
        if self.sample_outliers is not None and "flagged" in self.sample_outliers.columns:
            n_out = int(self.sample_outliers["flagged"].sum())
        n_mnar = 0
        mnar_str = "?"
        if self.feature_mechanism is not None and "mechanism" in self.feature_mechanism.columns:
            n_mnar = int((self.feature_mechanism["mechanism"] == "MNAR").sum())
            mnar_str = f"{n_mnar} MNAR features"
        else:
            mnar_str = "mechanism not run"
        return (
            f"MismapReport(n={self.n_features}x{self.n_samples}, "
            f"{n_out} outliers, {mnar_str}, passed={self.passed})"
        )

    def __str__(self) -> str:
        return self.summary()

    def summary(self) -> str:
        lines = ["MismapReport", "============"]
        lines.append(f"  Shape: {self.n_features} features x {self.n_samples} samples")
        if self.group_level is not None:
            lines.append(f"  Group level: {self.group_level}")

        if self.sample_missing_rate is not None:
            comp = 1 - self.sample_missing_rate
            lines.append("")
            lines.append("  Sample completeness")
            lines.append(f"    median: {comp.median():.3f}")
            lines.append(f"    range:  {comp.min():.3f} - {comp.max():.3f}")

        if self.sample_outliers is not None and "flagged" in self.sample_outliers.columns:
            flagged = self.sample_outliers[self.sample_outliers["flagged"]]
            lines.append(f"    outliers flagged: {len(flagged)}")
            if len(flagged):
                names = list(flagged["sample"].astype(str).head(5))
                suffix = ", ..." if len(flagged) > 5 else ""
                lines.append(f"      ({', '.join(names)}{suffix})")

        if self.feature_detection_rate is not None:
            d = self.feature_detection_rate
            lines.append("")
            lines.append("  Feature detection")
            n_full = int((d >= 0.9999).sum())
            pct_full = (d >= 0.9999).mean() * 100
            lines.append(f"    detected in all samples: {n_full} ({pct_full:.0f}%)")
            lines.append(f"    detected in <50%:        {int((d < 0.5).sum())}")

        if self.feature_mechanism is not None and "mechanism" in self.feature_mechanism.columns:
            lines.append("")
            lines.append("  Mechanism classification")
            counts = self.feature_mechanism["mechanism"].value_counts()
            for k, v in counts.items():
                lines.append(f"    {k}: {v}")

        lines.append("")
        verdict = "PASSED" if self.passed else "FAILED"
        if not self.results:
            lines.append(f"  Verdict: {verdict} (no thresholds specified)")
        else:
            lines.append(f"  Verdict: {verdict}")
            if self.errors:
                lines.append(f"    errors:   {len(self.errors)}")
            if self.warnings:
                lines.append(f"    warnings: {len(self.warnings)}")
        return "\n".join(lines)

class MismapQCFailure(AssertionError):
    """Raised by assert_qc when one or more error-severity QC rules fail."""

    def __init__(self, report: MismapReport):
        self.report = report
        super().__init__(self._format())

    def _format(self) -> str:
        errors = self.report.errors
        warns = self.report.warnings
        n_e = len(errors)
        n_w = len(warns)
        header = f"MismapQCFailure: {n_e} rule{'s' if n_e != 1 else ''} failed"
        if n_w:
            header += f" ({n_w} warning{'s' if n_w != 1 else ''} suppressed)"
        lines = [header, ""]
        for r in errors:
            lines.append(f"  [ERROR] {r.rule}")
            lines.append(f"    threshold: {r.threshold}")
            lines.append(f"    actual:    {r.actual:.4g}")
            if r.detail:
                lines.append(f"    {r.detail}")
            lines.append("")
        for r in warns:
            lines.append(f"  [WARNING] {r.rule}  (suppressed, did not raise)")
            lines.append(f"    threshold: {r.threshold}")
            lines.append(f"    actual:    {r.actual:.4g}")
            if r.detail:
                lines.append(f"    {r.detail}")
            lines.append("")
        return "\n".join(lines)

class _SkipRule(Exception):
    """Internal: raised when a rule cannot be evaluated (prerequisite check missing)."""

def _evaluate_thresholds(
    report: MismapReport,
    thresholds: dict | None,
    severity_overrides: dict | None = None,
    *,
    emit_warnings: bool = True,
) -> tuple:
    """Evaluate a dict of {rule_name: threshold} against a report.

    Returns a tuple of RuleResult. Emits MismapQCWarning for failed warning-severity
    rules when emit_warnings=True. Raises ValueError on unknown rule names.
    """
    if not thresholds:
        return ()
    unknown = set(thresholds) - set(_RULE_DEFAULT_SEVERITY)
    if unknown:
        raise ValueError(
            f"Unknown threshold rule(s): {sorted(unknown)}. "
            f"Known: {sorted(_RULE_DEFAULT_SEVERITY)}"
        )
    overrides = severity_overrides or {}
    unknown_sev = set(overrides) - set(_RULE_DEFAULT_SEVERITY)
    if unknown_sev:
        raise ValueError(f"Unknown rule in severity_overrides: {sorted(unknown_sev)}")
    for sev in overrides.values():
        if sev not in ("error", "warning", "info"):
            raise ValueError(
                f"Severity must be 'error', 'warning', or 'info'; got {sev!r}"
            )

    results = []
    for rule, threshold in thresholds.items():
        severity = overrides.get(rule, _RULE_DEFAULT_SEVERITY[rule])
        evaluator = _RULE_EVALUATORS[rule]
        try:
            results.append(evaluator(report, float(threshold), severity))
        except _SkipRule:
            continue

    out = tuple(results)
    if emit_warnings:
        for r in out:
            if r.severity == "warning" and not r.passed:
                _warnings.warn(
                    f"{r.rule}: threshold {r.threshold}, actual {r.actual:.4g}. {r.detail}",
                    MismapQCWarning,
                    stacklevel=3,
                )
    return out

def _rule_min_sample_completeness(report, threshold, severity):
    if report.sample_missing_rate is None:
        raise _SkipRule
    completeness = 1 - report.sample_missing_rate
    min_comp = float(completeness.min())
    passed = min_comp >= threshold
    detail = ""
    if not passed:
        bad = completeness[completeness < threshold].sort_values().head(5)
        detail = "offenders: " + ", ".join(f"{s} ({v:.2f})" for s, v in bad.items())
    return RuleResult("min_sample_completeness", severity, passed, threshold, min_comp, detail)

def _rule_min_sample_completeness_per_group(report, threshold, severity):
    if (
        report.sample_outliers is None
        or "group" not in report.sample_outliers.columns
        or report.sample_outliers["group"].isna().all()
    ):
        raise _SkipRule
    df = report.sample_outliers.copy()
    df["completeness"] = 1 - df["missing_rate"]
    by_group = df.groupby("group")["completeness"].min()
    worst = float(by_group.min())
    passed = worst >= threshold
    detail = ""
    if not passed:
        bad = by_group[by_group < threshold].sort_values().head(3)
        detail = "worst groups: " + ", ".join(f"{g} ({v:.2f})" for g, v in bad.items())
    return RuleResult(
        "min_sample_completeness_per_group", severity, passed, threshold, worst, detail
    )

def _rule_min_features_detected(report, threshold, severity):
    if report.feature_detection_rate is None:
        raise _SkipRule
    n_detected = int((report.feature_detection_rate > 0).sum())
    passed = n_detected >= threshold
    return RuleResult(
        "min_features_detected", severity, passed, threshold, float(n_detected), ""
    )

def _rule_max_feature_missing_rate(report, threshold, severity):
    if report.feature_detection_rate is None:
        raise _SkipRule
    miss = 1 - report.feature_detection_rate
    worst = float(miss.max())
    passed = worst <= threshold
    detail = ""
    if not passed:
        bad = miss[miss > threshold].sort_values(ascending=False).head(5)
        detail = "worst features: " + ", ".join(f"{f} ({v:.2f})" for f, v in bad.items())
    return RuleResult(
        "max_feature_missing_rate", severity, passed, threshold, worst, detail
    )

def _rule_max_sample_outliers(report, threshold, severity):
    if report.sample_outliers is None or "flagged" not in report.sample_outliers.columns:
        raise _SkipRule
    flagged = report.sample_outliers[report.sample_outliers["flagged"]]
    n = int(len(flagged))
    passed = n <= threshold
    detail = ""
    if not passed:
        names = list(flagged["sample"].astype(str).head(5))
        suffix = ", ..." if n > 5 else ""
        detail = f"flagged: {', '.join(names)}{suffix}"
    return RuleResult(
        "max_sample_outliers", severity, passed, threshold, float(n), detail
    )

def _rule_max_sample_outlier_zscore(report, threshold, severity):
    if report.sample_outliers is None or "z_score" not in report.sample_outliers.columns:
        raise _SkipRule
    z = report.sample_outliers["z_score"]
    max_z = float(z.abs().max())
    passed = max_z <= threshold
    detail = ""
    if not passed:
        worst = report.sample_outliers.iloc[
            z.abs().sort_values(ascending=False).index
        ].head(3)
        detail = "highest: " + ", ".join(
            f"{r['sample']} (z={r['z_score']:.2f})" for _, r in worst.iterrows()
        )
    return RuleResult(
        "max_sample_outlier_zscore", severity, passed, threshold, max_z, detail
    )

def _rule_max_mnar_fraction(report, threshold, severity):
    if report.feature_mechanism is None or "mechanism" not in report.feature_mechanism.columns:
        raise _SkipRule
    mech = report.feature_mechanism["mechanism"]
    testable = mech[mech != "INSUFFICIENT"]
    if len(testable) == 0:
        raise _SkipRule
    n_mnar = int((testable == "MNAR").sum())
    frac = float(n_mnar / len(testable))
    passed = frac <= threshold
    detail = f"{n_mnar} / {len(testable)} testable features"
    return RuleResult(
        "max_mnar_fraction", severity, passed, threshold, frac, detail
    )

def _rule_max_unclassified_fraction(report, threshold, severity):
    if report.feature_mechanism is None or "mechanism" not in report.feature_mechanism.columns:
        raise _SkipRule
    mech = report.feature_mechanism["mechanism"]
    frac = float((mech == "INSUFFICIENT").mean()) if len(mech) else 0.0
    passed = frac <= threshold
    return RuleResult(
        "max_unclassified_fraction", severity, passed, threshold, frac, ""
    )

def _rule_min_group_completeness(report, threshold, severity):
    if report.group_completeness is None:
        raise _SkipRule
    worst = float(report.group_completeness.min())
    passed = worst >= threshold
    detail = ""
    if not passed:
        bad = report.group_completeness[report.group_completeness < threshold]
        bad = bad.sort_values().head(3)
        detail = "worst groups: " + ", ".join(f"{g} ({v:.2f})" for g, v in bad.items())
    return RuleResult(
        "min_group_completeness", severity, passed, threshold, worst, detail
    )

def _rule_max_batch_effect_features(report, threshold, severity):
    if report.batch_test is None or "significant" not in report.batch_test.columns:
        raise _SkipRule
    n_sig = int(report.batch_test["significant"].sum())
    passed = n_sig <= threshold
    return RuleResult(
        "max_batch_effect_features", severity, passed, threshold, float(n_sig), ""
    )

def _rule_max_runorder_slope(report, threshold, severity):
    if report.runorder_trend is None or "slope" not in report.runorder_trend:
        raise _SkipRule
    slope = float(abs(report.runorder_trend["slope"]))
    passed = slope <= threshold
    return RuleResult(
        "max_runorder_slope", severity, passed, threshold, slope, ""
    )

_RULE_EVALUATORS = {
    "min_sample_completeness": _rule_min_sample_completeness,
    "min_sample_completeness_per_group": _rule_min_sample_completeness_per_group,
    "min_features_detected": _rule_min_features_detected,
    "max_feature_missing_rate": _rule_max_feature_missing_rate,
    "max_sample_outliers": _rule_max_sample_outliers,
    "max_sample_outlier_zscore": _rule_max_sample_outlier_zscore,
    "max_mnar_fraction": _rule_max_mnar_fraction,
    "max_unclassified_fraction": _rule_max_unclassified_fraction,
    "min_group_completeness": _rule_min_group_completeness,
    "max_batch_effect_features": _rule_max_batch_effect_features,
    "max_runorder_slope": _rule_max_runorder_slope,
}

def qc(
    df: pd.DataFrame,
    *,
    group_level: int | str | None = None,
    run_order=None,
    checks: tuple[str, ...] = ("completeness", "outliers", "mechanism", "codropouts"),
    thresholds: dict | None = None,
    severity_overrides: dict | None = None,
    feature_type: str = "PROT",
    verbose: bool = False,
) -> MismapReport:
    """Run a missing-data QC battery and return a MismapReport.

    Parameters
    ----------
    df : DataFrame
        features (rows) x samples (columns). NaN = missing.
    group_level : int or str, optional
        MultiIndex column level for per-group analyses. Required for the
        "batch" check and for per-group outlier z-scoring.
    run_order : array-like, optional
        Per-sample run order (numeric). Required for the "runorder" check.
    checks : tuple of str
        Which checks to run. Available: "completeness", "outliers", "mechanism",
        "codropouts", "batch", "runorder".
    thresholds : dict, optional
        Rule name -> threshold value. Triggers rule evaluation.
    severity_overrides : dict, optional
        Rule name -> "error" | "warning" | "info". Overrides defaults.
    feature_type : str
        "PROT" | "GENE" | "PEPTIDE". Used by downstream rendering.
    verbose : bool
        Print progress per check.

    Returns
    -------
    MismapReport

    Examples
    --------
    >>> report = qc(df)
    >>> report.passed
    >>> report = qc(df, group_level="Condition")
    >>> report.sample_outliers.query("flagged")
    >>> report = qc(df, thresholds={"min_sample_completeness": 0.6})
"""
    n_features, n_samples = df.shape
    _, groups = _resolve_group_labels(df, group_level)

    sample_missing_rate = None
    feature_detection_rate = None
    group_completeness = None
    sample_outliers = None
    feature_mechanism = None
    feature_codropouts = None
    batch_test = None
    runorder_trend = None

    if "completeness" in checks:
        if verbose:
            print("  completeness...")
        sample_missing_rate = df.isna().mean(axis=0)
        feature_detection_rate = df.notna().mean(axis=1)
        if groups is not None:
            completeness_per_sample = 1 - sample_missing_rate.values
            unique = pd.unique(groups)
            group_completeness = pd.Series(
                {
                    g: float(completeness_per_sample[groups == g].mean())
                    for g in unique
                },
                name="group_completeness",
            )

    if "outliers" in checks:
        if verbose:
            print("  outliers...")
        if sample_missing_rate is None:
            sample_missing_rate = df.isna().mean(axis=0)
        sample_outliers = _compute_sample_outliers(df, sample_missing_rate, groups)

    if "mechanism" in checks:
        if verbose:
            print("  mechanism...")
        feature_mechanism = _classify_mechanism(df)

    if "codropouts" in checks:
        if verbose:
            print("  codropouts...")
        feature_codropouts = _top_codropouts(df)

    if "batch" in checks and groups is not None:
        if verbose:
            print("  batch...")
        unique = sorted(pd.unique(groups).tolist())
        if len(unique) == 2:
            batch_test = _batch_missing_test(df, groups, unique[0], unique[1])

    if "runorder" in checks and run_order is not None:
        if verbose:
            print("  runorder...")
        if sample_missing_rate is None:
            sample_missing_rate = df.isna().mean(axis=0)
        runorder_trend = _runorder_trend(sample_missing_rate, run_order)

    report = MismapReport(
        n_features=n_features,
        n_samples=n_samples,
        group_level=group_level,
        sample_missing_rate=sample_missing_rate,
        sample_outliers=sample_outliers,
        feature_detection_rate=feature_detection_rate,
        feature_mechanism=feature_mechanism,
        feature_codropouts=feature_codropouts,
        group_completeness=group_completeness,
        batch_test=batch_test,
        runorder_trend=runorder_trend,
        results=(),
    )
    if thresholds:
        results = _evaluate_thresholds(
            report, thresholds, severity_overrides, emit_warnings=True
        )
        report = replace(report, results=results)
    return report

def assert_qc(
    df: pd.DataFrame,
    *,
    thresholds: dict,
    severity_overrides: dict | None = None,
    **qc_kwargs,
) -> MismapReport:
    """Run qc() and raise MismapQCFailure on any error-severity failure.

    Warnings still emit via the warnings module but do not raise. Returns the
    populated MismapReport on success so the caller can keep using it.

    Examples
    --------
    >>> assert_qc(df, thresholds={"min_sample_completeness": 0.1})
    >>> assert_qc(df, thresholds={"max_mnar_fraction": 0.9}, group_level="Condition")
"""
    report = qc(
        df,
        thresholds=thresholds,
        severity_overrides=severity_overrides,
        **qc_kwargs,
    )
    if not report.passed:
        raise MismapQCFailure(report)
    return report
