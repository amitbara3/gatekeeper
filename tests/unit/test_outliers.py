"""Outlier profiling -- reports, never trims (R1.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gatekeeper.checks.outliers import check_outlier_leverage, profile_metric
from gatekeeper.data.schema import ExperimentData
from gatekeeper.data.synthetic import make_cookie_cats_like
from gatekeeper.spec import OutlierRule
from gatekeeper.types import DataSource, InsufficientData


def _frame_with(control_values: list[int], treatment_values: list[int]) -> ExperimentData:
    n_c, n_t = len(control_values), len(treatment_values)
    df = pd.DataFrame(
        {
            "userid": range(1, n_c + n_t + 1),
            "version": ["gate_30"] * n_c + ["gate_40"] * n_t,
            "sum_gamerounds": control_values + treatment_values,
            "retention_1": [False] * (n_c + n_t),
            "retention_7": [False] * (n_c + n_t),
        }
    )
    return ExperimentData.from_frame(df, data_source=DataSource.SYNTHETIC)


class TestProfileKnownAnswers:
    def test_hand_computable_profile(self):
        data = _frame_with([1, 2, 3, 4], [1, 1, 1, 1])
        p = profile_metric(data, "sum_gamerounds", "gate_30")
        assert p.n == 4
        assert p.mean == pytest.approx(2.5)
        assert p.median == pytest.approx(2.5)
        assert p.minimum == 1.0
        assert p.maximum == 4.0
        # ddof=1: var = ((1.5)^2+(0.5)^2+(0.5)^2+(1.5)^2)/3 = 5/3
        assert p.std == pytest.approx(np.sqrt(5 / 3))

    def test_max_leverage_is_hand_computable(self):
        # [1, 1, 1, 97]: mean = 25. Without max: mean = 1. Shift = 24/25 = 96%.
        data = _frame_with([1, 1, 1, 97], [1, 1, 1, 1])
        p = profile_metric(data, "sum_gamerounds", "gate_30")
        assert p.mean == pytest.approx(25.0)
        assert p.max_leverage == pytest.approx(0.96)

    def test_uniform_data_has_near_zero_leverage(self):
        data = _frame_with([10] * 100, [10] * 100)
        p = profile_metric(data, "sum_gamerounds", "gate_30")
        assert p.max_leverage == pytest.approx(0.0, abs=1e-12)

    def test_skew_ratio_flags_a_pulled_mean(self):
        data = _frame_with([1, 1, 1, 1, 996], [1, 1, 1, 1, 1])
        p = profile_metric(data, "sum_gamerounds", "gate_30")
        assert p.median == 1.0
        assert p.mean == pytest.approx(200.0)
        assert p.skew_ratio == pytest.approx(200.0)

    def test_skew_ratio_handles_zero_median(self):
        data = _frame_with([0, 0, 0, 8], [1, 1, 1, 1])
        p = profile_metric(data, "sum_gamerounds", "gate_30")
        assert p.median == 0.0
        assert p.skew_ratio == float("inf")

    def test_all_zero_metric_gives_zero_top_share(self):
        data = _frame_with([0, 0, 0, 0], [0, 0, 0, 0])
        p = profile_metric(data, "sum_gamerounds", "gate_30")
        assert p.top_share == 0.0
        assert p.max_leverage == 0.0

    def test_top_share_is_hand_computable(self):
        # 999 ones and one 1001: total 2000, top unit holds 1001/2000.
        data = _frame_with([1] * 999 + [1001], [1] * 1000)
        p = profile_metric(data, "sum_gamerounds", "gate_30")
        assert p.top_share == pytest.approx(1001 / 2000)

    def test_single_observation_arm_raises(self):
        data = _frame_with([5], [1, 2, 3])
        with pytest.raises(InsufficientData, match=">= 2 observations"):
            profile_metric(data, "sum_gamerounds", "gate_30")

    def test_describe_is_readable(self):
        data = _frame_with([1, 2, 3, 400], [1, 1, 1, 1])
        text = profile_metric(data, "sum_gamerounds", "gate_30").describe()
        assert "sum_gamerounds" in text
        assert "gate_30" in text
        assert "mean" in text


class TestBinaryMetricGuard:
    """A binary metric must be refused, not answered plausibly.

    Pointed at retention_7 this module would compute a tiny leverage, report "no
    single unit dominates the mean", and pass -- arithmetically true and analytically
    empty. Confident nonsense for an inappropriate input is exactly what this project
    exists to catch.
    """

    def test_profile_rejects_a_boolean_metric(self):
        data = _frame_with([1, 2, 3, 4], [1, 1, 1, 1])
        with pytest.raises(ValueError, match="binary metric"):
            profile_metric(data, "retention_7", "gate_30")

    def test_error_points_at_the_right_tool(self):
        data = _frame_with([1, 2, 3, 4], [1, 1, 1, 1])
        with pytest.raises(ValueError, match="two-proportion test"):
            profile_metric(data, "retention_1", "gate_30")

    def test_magnitude_metric_is_still_allowed(self):
        data = _frame_with([1, 2, 3, 4], [1, 1, 1, 1])
        assert profile_metric(data, "sum_gamerounds", "gate_30").n == 4

    def test_column_absent_from_the_schema_is_not_second_guessed(self):
        """An undeclared column has no declared kind; do not invent a rule for it."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "userid": [1, 2, 3, 4],
                "version": ["gate_30"] * 2 + ["gate_40"] * 2,
                "sum_gamerounds": [1, 2, 3, 4],
                "retention_1": [True] * 4,
                "retention_7": [False] * 4,
            }
        )
        data = ExperimentData.from_frame(df, data_source=DataSource.SYNTHETIC)
        # 'derived' is not in COOKIE_CATS, so the guard stays out of the way and the
        # real failure surfaces from the accessor instead.
        with pytest.raises(KeyError, match="no column"):
            profile_metric(data, "derived", "gate_30")


class TestLeverageCheck:
    def test_wellbehaved_metric_passes(self):
        data = _frame_with([10] * 500, [10] * 500)
        check = check_outlier_leverage(data, "sum_gamerounds")
        assert check.passed

    def test_dominated_mean_with_no_declared_rule_fails(self):
        data = _frame_with([1] * 99 + [10_000], [1] * 100)
        check = check_outlier_leverage(data, "sum_gamerounds", leverage_threshold=0.01)
        assert not check.passed
        assert "NO rule was pre-declared" in check.detail
        assert "R1.6" in check.detail

    def test_dominated_mean_with_a_declared_rule_passes(self):
        """A heavy tail is a metric property, not a defect -- if it was planned for.

        Blocking a readout because rounds-played is lognormal would be wrong. What
        must not happen is meeting the tail for the first time during analysis.
        """
        data = _frame_with([1] * 99 + [10_000], [1] * 100)
        rule = OutlierRule(metric="sum_gamerounds", method="winsorize", percentile=99.9)
        check = check_outlier_leverage(
            data, "sum_gamerounds", leverage_threshold=0.01, declared_rule=rule
        )
        assert check.passed
        assert "spec anticipated this" in check.detail
        assert "with and without it" in check.detail

    def test_a_none_rule_does_not_satisfy_the_check(self):
        """Declaring 'no trimming' is not the same as planning for a heavy tail."""
        data = _frame_with([1] * 99 + [10_000], [1] * 100)
        rule = OutlierRule(metric="sum_gamerounds", method="none")
        check = check_outlier_leverage(
            data, "sum_gamerounds", leverage_threshold=0.01, declared_rule=rule
        )
        assert not check.passed

    def test_declared_rule_does_not_mask_a_well_behaved_metric(self):
        data = _frame_with([10] * 500, [10] * 500)
        rule = OutlierRule(metric="sum_gamerounds", method="winsorize", percentile=99.9)
        check = check_outlier_leverage(data, "sum_gamerounds", declared_rule=rule)
        assert check.passed
        assert "no single unit dominates" in check.detail

    def test_check_picks_the_worst_arm(self):
        data = _frame_with([1] * 100, [1] * 99 + [10_000])
        check = check_outlier_leverage(data, "sum_gamerounds", leverage_threshold=0.01)
        assert not check.passed
        assert "gate_40" in check.detail

    def test_threshold_is_recorded(self):
        data = _frame_with([10] * 200, [10] * 200)
        assert (
            check_outlier_leverage(data, "sum_gamerounds", leverage_threshold=0.02).threshold
            == 0.02
        )

    def test_rejects_nonsense_threshold(self):
        data = _frame_with([10] * 200, [10] * 200)
        with pytest.raises(ValueError, match="must be positive"):
            check_outlier_leverage(data, "sum_gamerounds", leverage_threshold=0.0)

    def test_binary_metric_is_rejected(self):
        """Outlier concepts do not apply to a Bernoulli metric."""
        data = _frame_with([1, 2, 3, 400], [1, 1, 1, 1])
        with pytest.raises(ValueError, match="binary metric"):
            check_outlier_leverage(data, "retention_7")

    def test_never_mutates_the_data(self):
        """R1.6: this module reports; it does not trim."""
        exp = make_cookie_cats_like(n=2_000, seed=5)
        before = exp.data.frame.copy()
        check_outlier_leverage(exp.data, "sum_gamerounds", leverage_threshold=1e-9)
        profile_metric(exp.data, "sum_gamerounds", "gate_30")
        pd.testing.assert_frame_equal(before, exp.data.frame)
