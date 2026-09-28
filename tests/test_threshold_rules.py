"""Behaviour of the 11 threshold rules, on hand-built data.

Every expected value in this file is derived from the data by reading it, not by
running the code. That is the point: a test whose expected value came from the
code under test passes even when the code is wrong.

Each rule is checked four ways:

- its computed value (``actual``) matches the hand-derived answer
- it passes on one side of the threshold and fails on the other
- a value exactly at the threshold passes, because thresholds are inclusive
- it is skipped, rather than reported, when the check it depends on did not run

``test_validation_api.py`` covers the machinery around rules: severity
overrides, unknown rule names, ``assert_qc`` raising. This file covers what each
rule computes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mismap_qc import qc

nan = np.nan
ALL_CHECKS = ("completeness", "outliers", "mechanism", "codropouts", "batch", "runorder")


@pytest.fixture(autouse=True)
def _quiet_warning_rules():
    # Failing warning-severity rules emit MismapQCWarning. Expected here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _cols(samples, groups):
    return pd.MultiIndex.from_tuples(list(zip(groups, samples)), names=["group", "sample"])


# --- hand-built data ------------------------------------------------------


def completeness_data() -> pd.DataFrame:
    """Two groups of two samples. B_0 is missing 2 of its 4 values.

    sample completeness   A_0 1.0, A_1 1.0, B_0 0.5, B_1 1.0      min 0.5
    feature missing rate  f1 and f2 each miss 1 of 4 samples      max 0.25
    group completeness    A mean(1.0, 1.0) = 1.0
                          B mean(0.5, 1.0) = 0.75                 min 0.75
    """
    return pd.DataFrame(
        [[10, 11, 12, 9],
         [20, 21, nan, 19],
         [30, 31, nan, 29],
         [40, 41, 42, 39]],
        index=["f0", "f1", "f2", "f3"],
        columns=_cols(["A_0", "A_1", "B_0", "B_1"], ["A", "A", "B", "B"]),
        dtype=float,
    )


def detected_data() -> pd.DataFrame:
    """Three features, one of which is never measured in any sample.

    A feature counts as detected if it is present in at least one sample,
    so 2 of the 3 are detected. The all-missing row is what makes this test
    distinguish "count detected features" from "count all features".
    """
    return pd.DataFrame(
        [[1, 2, 3],
         [nan, nan, nan],
         [4, nan, 6]],
        index=["f0", "never", "f2"],
        columns=["s0", "s1", "s2"],
        dtype=float,
    )


def outlier_data(n: int = 9) -> pd.DataFrame:
    """n samples. The last is missing 2 of 4 values, the rest are complete.

    For one value among n-1 identical ones, the sample z-score (ddof=1) is
    exactly (n - 1) / sqrt(n), whatever the value itself. That is also the
    largest |z| any sample can reach with n samples.

    n = 9: z = 8 / 3 = 2.667, above the 2.5 cutoff, so the last sample is flagged.
    """
    df = pd.DataFrame(np.ones((4, n)), columns=[f"s{j}" for j in range(n)])
    df.iloc[:2, n - 1] = nan
    return df


def mechanism_data() -> pd.DataFrame:
    """Eight samples. Every value in column j is 10 + 10j.

    Constant columns make each sample's mean abundance exactly 10 + 10j however
    many values are missing, so the Mann-Whitney comparison is fully determined.

    mnar     missing in s0-s3, the four lowest-abundance samples. Every present
             sample outranks every absent one: exact one-sided p = 1/70 = 0.014,
             below 0.05, so MNAR.
    mar      missing in s1, s3, s5, s7, interleaved with the present ones: MAR.
    c1-c3    complete, so no absent samples to compare: INSUFFICIENT.

    MNAR fraction          1 MNAR of 2 testable (INSUFFICIENT excluded)   0.5
    unclassified fraction  3 INSUFFICIENT of 5 features                   0.6
    """
    values = np.array([10 + 10 * j for j in range(8)], dtype=float)
    df = pd.DataFrame(
        np.tile(values, (5, 1)),
        index=["mnar", "mar", "c1", "c2", "c3"],
        columns=[f"s{j}" for j in range(8)],
    )
    df.loc["mnar", ["s0", "s1", "s2", "s3"]] = nan
    df.loc["mar", ["s1", "s3", "s5", "s7"]] = nan
    return df


def batch_data() -> pd.DataFrame:
    """Two groups of five samples.

    lost     missing in all five B samples, present in all five A.
             Fisher's exact on [[0, 5], [5, 0]]: p = 2/252 = 0.0079.
    noise    missing once in each group. [[1, 4], [1, 4]]: p = 1.0.
    c1, c2   complete. Fewer than two missing values, so not tested at all.

    Two features tested. q = p * m / rank gives lost 0.016, noise 1.0,
    so exactly one feature is significant.
    """
    samples = [f"A_{i}" for i in range(5)] + [f"B_{i}" for i in range(5)]
    df = pd.DataFrame(
        np.ones((4, 10)),
        index=["lost", "noise", "c1", "c2"],
        columns=_cols(samples, ["A"] * 5 + ["B"] * 5),
    )
    df.iloc[0, 5:] = nan
    df.iloc[1, [0, 5]] = nan
    return df


def runorder_data() -> pd.DataFrame:
    """Five samples, where sample j is missing j of 4 features.

    Missing rates 0, 0.25, 0.5, 0.75, 1.0 against run order 1-5: an exact
    straight line with slope 0.25.
    """
    df = pd.DataFrame(np.ones((4, 5)), columns=[f"s{j}" for j in range(5)])
    for j in range(5):
        df.iloc[:j, j] = nan
    return df


# --- reports ---------------------------------------------------------------


def completeness_report():
    return qc(completeness_data(), group_level="group")


def detected_report():
    return qc(detected_data())


def outlier_report():
    return qc(outlier_data())


def mechanism_report():
    return qc(mechanism_data())


def batch_report():
    return qc(batch_data(), group_level="group", checks=ALL_CHECKS)


def runorder_report():
    return qc(runorder_data(), run_order=[1, 2, 3, 4, 5], checks=ALL_CHECKS)


def _result(report, rule, threshold):
    """The RuleResult for `rule`, or None if the rule was skipped."""
    matches = [r for r in report.check({rule: threshold}) if r.rule == rule]
    return matches[0] if matches else None


# --- the table ---------------------------------------------------------------
#
# min_ rules pass when actual >= threshold:  pass < actual < fail
# max_ rules pass when actual <= threshold:  fail < actual < pass

RULES = [
    # builder              rule                                  actual  pass  fail
    (completeness_report, "min_sample_completeness",             0.50,  0.40, 0.60),
    (completeness_report, "min_sample_completeness_per_group",   0.50,  0.40, 0.60),
    (detected_report,     "min_features_detected",               2,     1,    3),
    (completeness_report, "max_feature_missing_rate",            0.25,  0.30, 0.20),
    (outlier_report,      "max_sample_outliers",                 1,     2,    0),
    (outlier_report,      "max_sample_outlier_zscore",           8 / 3, 3.0,  2.0),
    (mechanism_report,    "max_mnar_fraction",                   0.50,  0.60, 0.40),
    (mechanism_report,    "max_unclassified_fraction",           0.60,  0.70, 0.50),
    (completeness_report, "min_group_completeness",              0.75,  0.70, 0.80),
    (batch_report,        "max_batch_effect_features",           1,     2,    0),
    (runorder_report,     "max_runorder_slope",                  0.25,  0.30, 0.20),
]
RULE_IDS = [row[1] for row in RULES]


@pytest.mark.parametrize("build, rule, actual, pass_at, fail_at", RULES, ids=RULE_IDS)
def test_actual_matches_hand_derived_value(build, rule, actual, pass_at, fail_at):
    result = _result(build(), rule, pass_at)
    assert result is not None, f"{rule} was skipped"
    assert result.actual == pytest.approx(actual)


@pytest.mark.parametrize("build, rule, actual, pass_at, fail_at", RULES, ids=RULE_IDS)
def test_passes_on_the_right_side_of_the_threshold(build, rule, actual, pass_at, fail_at):
    assert _result(build(), rule, pass_at).passed is True


@pytest.mark.parametrize("build, rule, actual, pass_at, fail_at", RULES, ids=RULE_IDS)
def test_fails_on_the_wrong_side_of_the_threshold(build, rule, actual, pass_at, fail_at):
    assert _result(build(), rule, fail_at).passed is False


@pytest.mark.parametrize("build, rule, actual, pass_at, fail_at", RULES, ids=RULE_IDS)
def test_threshold_is_inclusive(build, rule, actual, pass_at, fail_at):
    # Use the rule's own computed value as the threshold. That tests the
    # comparison (>= or <=) without depending on the value being exactly
    # representable as a float, which is test_actual_matches_hand_derived_value's job.
    report = build()
    at = _result(report, rule, pass_at).actual
    assert _result(report, rule, at).passed is True


# --- failure messages ------------------------------------------------------

OFFENDERS = [
    # builder              rule                                  fail  names  not
    (completeness_report, "min_sample_completeness",             0.60, "B_0", "A_0"),
    (completeness_report, "min_sample_completeness_per_group",   0.60, "B",   "A"),
    (completeness_report, "max_feature_missing_rate",            0.20, "f1",  "f0"),
    (outlier_report,      "max_sample_outliers",                 0,    "s8",  "s0"),
    # Lists the three highest |z| samples whether or not they cross the
    # threshold, so only check the real outlier is named, not that others are absent.
    (outlier_report,      "max_sample_outlier_zscore",           2.0,  "s8",  None),
    (completeness_report, "min_group_completeness",              0.80, "B",   "A"),
]


@pytest.mark.parametrize(
    "build, rule, fail_at, named, not_named", OFFENDERS, ids=[r[1] for r in OFFENDERS]
)
def test_failure_names_the_offender(build, rule, fail_at, named, not_named):
    detail = _result(build(), rule, fail_at).detail
    assert named in detail
    if not_named is not None:
        assert not_named not in detail


# --- skipping --------------------------------------------------------------
#
# A rule whose prerequisite check did not run must be absent from the results,
# not reported as a pass. A silent pass would tell a pipeline its data is fine
# when nothing was measured.

SKIPS = [
    pytest.param(lambda: qc(completeness_data(), checks=("mechanism",)),
                 "min_sample_completeness", id="completeness-not-run"),
    pytest.param(outlier_report,
                 "min_sample_completeness_per_group", id="per-group-without-groups"),
    pytest.param(lambda: qc(completeness_data(), checks=("mechanism",)),
                 "min_features_detected", id="detected-without-completeness"),
    pytest.param(lambda: qc(completeness_data(), checks=("mechanism",)),
                 "max_feature_missing_rate", id="missing-rate-without-completeness"),
    pytest.param(lambda: qc(completeness_data(), checks=("completeness",)),
                 "max_sample_outliers", id="outliers-not-run"),
    pytest.param(lambda: qc(completeness_data(), checks=("completeness",)),
                 "max_sample_outlier_zscore", id="zscore-without-outliers"),
    pytest.param(completeness_report,
                 "max_mnar_fraction", id="mnar-with-nothing-testable"),
    pytest.param(lambda: qc(completeness_data(), checks=("completeness",)),
                 "max_unclassified_fraction", id="mechanism-not-run"),
    pytest.param(detected_report,
                 "min_group_completeness", id="group-completeness-without-groups"),
    pytest.param(lambda: qc(batch_data(), group_level="group"),
                 "max_batch_effect_features", id="batch-not-in-default-checks"),
    pytest.param(lambda: qc(runorder_data(), checks=ALL_CHECKS),
                 "max_runorder_slope", id="runorder-without-run-order"),
]


@pytest.mark.parametrize("build, rule", SKIPS)
def test_rule_is_skipped_when_its_check_did_not_run(build, rule):
    assert _result(build(), rule, 0.5) is None


# --- properties of the statistics ------------------------------------------


def test_runorder_slope_is_absolute():
    """A falling trend is as much drift as a rising one."""
    report = qc(runorder_data(), run_order=[5, 4, 3, 2, 1], checks=ALL_CHECKS)
    assert _result(report, "max_runorder_slope", 0.3).actual == pytest.approx(0.25)


def test_mnar_is_undetectable_with_three_samples_on_each_side():
    """Three present against three absent can never be classified MNAR.

    Even perfectly separated, the smallest one-sided Mann-Whitney p-value for
    3 vs 3 is 1 / C(6, 3) = 1/20 = 0.05, which does not pass p < 0.05. So a
    feature missing in exactly the three lowest-abundance samples of six is
    called MAR.
    """
    values = np.array([10 + 10 * j for j in range(6)], dtype=float)
    df = pd.DataFrame(np.tile(values, (2, 1)), index=["separated", "c"],
                      columns=[f"s{j}" for j in range(6)])
    df.loc["separated", ["s0", "s1", "s2"]] = nan
    row = qc(df).feature_mechanism.set_index("feature").loc["separated"]
    assert row["p_value"] == pytest.approx(1 / 20)
    assert row["mechanism"] == "MAR"


@pytest.mark.parametrize("n, flagged", [(8, 0), (9, 1)], ids=["8-samples", "9-samples"])
def test_outlier_flagging_needs_at_least_nine_samples(n, flagged):
    """With n samples no |z| can exceed (n - 1) / sqrt(n).

    That bound stays below the 2.5 cutoff until n = 9: 2.47 at n = 8, 2.67 at
    n = 9. So with eight or fewer samples, however extreme one of them is,
    nothing is flagged.
    """
    report = qc(outlier_data(n))
    assert int(report.sample_outliers["flagged"].sum()) == flagged
