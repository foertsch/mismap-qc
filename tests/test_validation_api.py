"""Tests for the validation API: qc(), assert_qc(), MismapReport, RuleResult."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from mismap_qc import (  # noqa: E402
    MismapQCFailure,
    MismapQCWarning,
    MismapReport,
    RuleResult,
    assert_qc,
    qc,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_df(n_features=120, n_samples=20, missing_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(10, 2, size=(n_features, n_samples))
    mask = rng.random((n_features, n_samples)) < missing_frac
    data[mask] = np.nan
    return pd.DataFrame(
        data,
        index=[f"F{i}" for i in range(n_features)],
        columns=[f"S{i}" for i in range(n_samples)],
    )


def _make_multiindex_df(n_features=120, n_per_group=10, missing_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    conditions = ["A", "B"]
    tuples = [(c, f"S{c}{i}") for c in conditions for i in range(n_per_group)]
    cols = pd.MultiIndex.from_tuples(tuples, names=["condition", "sample"])
    data = rng.normal(10, 2, size=(n_features, len(tuples)))
    mask = rng.random((n_features, len(tuples))) < missing_frac
    data[mask] = np.nan
    return pd.DataFrame(data, index=[f"F{i}" for i in range(n_features)], columns=cols)


# ── MismapReport basics ───────────────────────────────────────────────────────


def test_qc_returns_report():
    report = qc(_make_df())
    assert isinstance(report, MismapReport)
    assert report.n_features == 120
    assert report.n_samples == 20


def test_report_frozen():
    report = qc(_make_df())
    with pytest.raises(Exception):  # FrozenInstanceError
        report.n_features = 999


def test_report_no_thresholds_passes():
    report = qc(_make_df())
    assert report.passed is True
    assert report.errors == ()
    assert report.warnings == ()
    assert report.results == ()


def test_repr_one_line():
    report = qc(_make_df())
    r = repr(report)
    assert r.startswith("MismapReport(")
    assert "\n" not in r
    assert "passed=" in r


def test_summary_multiline():
    report = qc(_make_df())
    s = report.summary()
    assert "MismapReport" in s
    assert "Shape:" in s
    assert "Verdict:" in s
    assert "\n" in s


def test_str_is_summary():
    report = qc(_make_df())
    assert str(report) == report.summary()


# ── qc() checks populate expected fields ─────────────────────────────────────


def test_qc_completeness_check():
    report = qc(_make_df(), checks=("completeness",))
    assert report.sample_missing_rate is not None
    assert report.feature_detection_rate is not None
    assert len(report.sample_missing_rate) == 20
    assert len(report.feature_detection_rate) == 120
    # Other fields untouched
    assert report.sample_outliers is None
    assert report.feature_mechanism is None


def test_qc_outliers_check():
    report = qc(_make_df(), checks=("outliers",))
    assert report.sample_outliers is not None
    assert set(report.sample_outliers.columns) >= {
        "sample", "group", "missing_rate", "z_score", "flagged",
    }
    assert len(report.sample_outliers) == 20


def test_qc_mechanism_check():
    report = qc(_make_df(), checks=("mechanism",))
    assert report.feature_mechanism is not None
    assert set(report.feature_mechanism.columns) >= {
        "feature", "mechanism", "missing_rate", "mean_abundance", "p_value",
    }
    assert set(report.feature_mechanism["mechanism"].unique()) <= {
        "MNAR", "MAR", "MCAR", "INSUFFICIENT",
    }


def test_qc_codropouts_check():
    report = qc(_make_df(), checks=("codropouts",))
    assert report.feature_codropouts is not None
    assert list(report.feature_codropouts.columns) == [
        "feature_a", "feature_b", "comissingness",
    ]


def test_qc_multiindex_group_level():
    df = _make_multiindex_df()
    report = qc(df, group_level="condition")
    assert report.group_completeness is not None
    assert len(report.group_completeness) == 2
    assert set(report.group_completeness.index) == {"A", "B"}


def test_qc_batch_check_requires_group_level():
    df = _make_multiindex_df()
    report = qc(df, group_level="condition", checks=("batch",))
    assert report.batch_test is not None
    assert {"feature", "log2_OR", "p_value", "q_value", "significant", "enriched_in"} <= set(
        report.batch_test.columns
    )


def test_qc_runorder_check():
    df = _make_df()
    run_order = list(range(20))
    report = qc(df, run_order=run_order, checks=("runorder",))
    assert report.runorder_trend is not None
    assert "slope" in report.runorder_trend
    assert "p" in report.runorder_trend


def test_qc_group_level_without_multiindex_raises():
    with pytest.raises(ValueError, match="MultiIndex"):
        qc(_make_df(), group_level="condition")


# ── Threshold evaluation ─────────────────────────────────────────────────────


def test_thresholds_pass_when_lenient():
    report = qc(_make_df(), thresholds={"min_sample_completeness": 0.1})
    assert report.passed is True
    assert len(report.results) == 1
    assert report.results[0].passed is True
    assert report.results[0].rule == "min_sample_completeness"


def test_thresholds_fail_when_strict():
    report = qc(_make_df(), thresholds={"min_sample_completeness": 0.99})
    assert report.passed is False
    assert len(report.errors) == 1
    assert report.errors[0].rule == "min_sample_completeness"
    assert report.errors[0].severity == "error"
    assert "offenders" in report.errors[0].detail


def test_warning_rule_does_not_block_passed():
    # max_feature_missing_rate defaults to warning severity
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        report = qc(_make_df(), thresholds={"max_feature_missing_rate": 0.01})
    assert report.passed is True  # warnings do not flip passed
    assert len(report.warnings) >= 1
    # MismapQCWarning emitted
    assert any(issubclass(ww.category, MismapQCWarning) for ww in w)


def test_unknown_rule_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown threshold rule"):
        qc(_make_df(), thresholds={"nonexistent_rule": 0.5})


def test_unknown_severity_override_raises():
    with pytest.raises(ValueError, match="Unknown rule in severity_overrides"):
        qc(
            _make_df(),
            thresholds={"min_sample_completeness": 0.5},
            severity_overrides={"nonexistent_rule": "warning"},
        )


def test_invalid_severity_value_raises():
    with pytest.raises(ValueError, match="Severity must be"):
        qc(
            _make_df(),
            thresholds={"min_sample_completeness": 0.5},
            severity_overrides={"min_sample_completeness": "critical"},
        )


def test_severity_override_downgrades_error_to_warning():
    df = _make_df()
    # Without override: error rule fails => raises
    with pytest.raises(MismapQCFailure):
        assert_qc(df, thresholds={"min_sample_completeness": 0.99})
    # With override: downgraded => does not raise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MismapQCWarning)
        report = assert_qc(
            df,
            thresholds={"min_sample_completeness": 0.99},
            severity_overrides={"min_sample_completeness": "warning"},
        )
    assert report.passed is True
    assert len(report.warnings) == 1


def test_assert_qc_raises_on_error():
    df = _make_df()
    with pytest.raises(MismapQCFailure) as exc_info:
        assert_qc(df, thresholds={"min_sample_completeness": 0.99})
    msg = str(exc_info.value)
    assert "MismapQCFailure" in msg
    assert "[ERROR]" in msg
    assert "min_sample_completeness" in msg
    # Report attached to exception
    assert isinstance(exc_info.value.report, MismapReport)


def test_assert_qc_returns_report_on_success():
    df = _make_df()
    report = assert_qc(df, thresholds={"min_sample_completeness": 0.1})
    assert isinstance(report, MismapReport)
    assert report.passed is True


def test_report_check_no_raise():
    report = qc(_make_df())
    # Even a failing threshold does not raise via check()
    results = report.check({"min_sample_completeness": 0.99})
    assert len(results) == 1
    assert results[0].passed is False
    # Original report results untouched
    assert report.results == ()


def test_report_passes_method():
    report = qc(_make_df())
    assert report.passes({"min_sample_completeness": 0.1}) is True
    assert report.passes({"min_sample_completeness": 0.99}) is False


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_qc_all_missing():
    df = pd.DataFrame(
        np.full((10, 5), np.nan),
        index=[f"F{i}" for i in range(10)],
        columns=[f"S{i}" for i in range(5)],
    )
    report = qc(df)
    assert isinstance(report, MismapReport)
    # All samples have 100% missing
    assert (report.sample_missing_rate == 1.0).all()


def test_qc_all_present():
    df = pd.DataFrame(
        np.ones((10, 5)),
        index=[f"F{i}" for i in range(10)],
        columns=[f"S{i}" for i in range(5)],
    )
    report = qc(df)
    assert isinstance(report, MismapReport)
    assert (report.sample_missing_rate == 0.0).all()
    # mechanism: all features have 0 missing => all INSUFFICIENT
    assert (report.feature_mechanism["mechanism"] == "INSUFFICIENT").all()


# ── Serialization ────────────────────────────────────────────────────────────


def test_to_dict_keys():
    report = qc(_make_df())
    d = report.to_dict()
    expected = {
        "n_features", "n_samples", "group_level",
        "sample_missing_rate", "sample_outliers",
        "feature_detection_rate", "feature_mechanism", "feature_codropouts",
        "group_completeness", "batch_test", "runorder_trend",
        "results", "passed",
    }
    assert set(d.keys()) == expected


def test_to_json_parses_back(tmp_path):
    report = qc(_make_df())
    s = report.to_json()
    parsed = json.loads(s)
    assert parsed["n_features"] == 120

    # With path
    out = tmp_path / "report.json"
    report.to_json(path=str(out))
    assert out.exists()
    assert json.loads(out.read_text())["n_samples"] == 20


def test_to_html_contains_verdict(tmp_path):
    report = qc(_make_df(), thresholds={"min_sample_completeness": 0.1})
    html = report.to_html()
    assert "MismapReport" in html
    assert ("PASSED" in html) or ("FAILED" in html)
    assert "min_sample_completeness" in html

    out = tmp_path / "r.html"
    report.to_html(path=str(out))
    assert out.exists()


# ── RuleResult dataclass ─────────────────────────────────────────────────────


def test_rule_result_frozen():
    r = RuleResult("rule_x", "error", True, 0.5, 0.6, "")
    with pytest.raises(Exception):
        r.rule = "other"


# ── Skipped rules ─────────────────────────────────────────────────────────────


def test_rule_skipped_when_check_not_run():
    # min_group_completeness requires the "completeness" check + group_level.
    # If group_level is None, group_completeness is None -> rule is silently skipped.
    report = qc(_make_df(), thresholds={"min_group_completeness": 0.5})
    # Rule was skipped; no result added; passed remains True.
    assert report.passed is True
    assert len(report.results) == 0
