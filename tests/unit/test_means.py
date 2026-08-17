"""Welch's t-test, with a fully hand-computed fixture."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gatekeeper.data.synthetic import make_cookie_cats_like
from gatekeeper.frequentist.means import estimate_welch, welch_test
from gatekeeper.types import Estimand, InsufficientData, Scale

CONTROL = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
TREATMENT = np.array([3.0, 4.0, 5.0, 6.0, 7.0])


class TestHandComputed:
    """control = [1..5], treatment = [3..7].

    mean_c = 3, mean_t = 5, diff = 2
    var (ddof=1) = 10/4 = 2.5 in both arms
    se = sqrt(2.5/5 + 2.5/5) = 1.0
    t = 2 / 1 = 2
    df = (0.5+0.5)^2 / (0.5^2/4 + 0.5^2/4) = 1 / 0.125 = 8
    """

    def test_point_and_se(self):
        r = welch_test(CONTROL, TREATMENT)
        assert r.mean_control == pytest.approx(3.0)
        assert r.mean_treatment == pytest.approx(5.0)
        assert r.point == pytest.approx(2.0)
        assert r.se == pytest.approx(1.0)

    def test_t_and_df(self):
        r = welch_test(CONTROL, TREATMENT)
        assert r.t == pytest.approx(2.0)
        # Equal variances and equal n collapse Welch's df to n1 + n2 - 2.
        assert r.df == pytest.approx(8.0)

    def test_p_value(self):
        from scipy import stats

        r = welch_test(CONTROL, TREATMENT)
        assert r.p_value == pytest.approx(2 * stats.t.sf(2.0, 8))

    def test_confidence_interval(self):
        from scipy import stats

        r = welch_test(CONTROL, TREATMENT)
        t_crit = stats.t.isf(0.025, 8)
        assert r.ci[0] == pytest.approx(2.0 - t_crit)
        assert r.ci[1] == pytest.approx(2.0 + t_crit)
        # Interval includes zero at alpha=0.05, matching p ~ 0.08.
        assert r.ci[0] < 0 < r.ci[1]

    def test_identical_arms_give_zero_and_p_one(self):
        r = welch_test(CONTROL, CONTROL.copy())
        assert r.point == pytest.approx(0.0)
        assert r.p_value == pytest.approx(1.0)


class TestUnequalVariance:
    def test_welch_df_drops_below_pooled_when_variances_differ(self):
        tight = np.array([5.0, 5.1, 4.9, 5.0, 5.1, 4.9, 5.0, 5.0])
        wide = np.array([1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0])
        r = welch_test(tight, wide)
        assert r.df < len(tight) + len(wide) - 2

    def test_unequal_variance_is_not_pooled(self):
        """A pooled test would give a different SE; confirm we use the Welch form."""
        a = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
        b = np.array([2.0, 2.0, 2.0, 2.0, 2.0000001])
        r = welch_test(a, b)
        expected_se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        assert r.se == pytest.approx(expected_se)


class TestDegenerate:
    def test_single_observation_arm_raises(self):
        with pytest.raises(InsufficientData, match=">= 2 observations"):
            welch_test(np.array([1.0]), TREATMENT)

    def test_both_arms_constant_and_identical(self):
        r = welch_test(np.ones(5), np.ones(5))
        assert r.se == 0.0
        assert r.point == 0.0
        assert r.p_value == 1.0

    def test_both_arms_constant_but_different(self):
        """A deterministic gap: report it, do not pretend a t-test applies."""
        r = welch_test(np.ones(5), np.full(5, 2.0))
        assert r.se == 0.0
        assert r.point == pytest.approx(1.0)
        assert r.p_value == 0.0
        assert r.ci == (1.0, 1.0)

    def test_bad_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            welch_test(CONTROL, TREATMENT, alpha=1.5)


class TestProperties:
    @settings(max_examples=150, deadline=None)
    @given(
        shift=st.floats(min_value=-50, max_value=50, allow_nan=False),
        scale=st.floats(min_value=0.5, max_value=20, allow_nan=False),
    )
    def test_swapping_arms_flips_sign(self, shift: float, scale: float):
        rng = np.random.default_rng(0)
        a = rng.normal(0, scale, 40)
        b = rng.normal(shift, scale, 40)
        fwd = welch_test(a, b)
        rev = welch_test(b, a)
        assert fwd.point == pytest.approx(-rev.point)
        assert fwd.p_value == pytest.approx(rev.p_value)
        assert fwd.df == pytest.approx(rev.df)
        assert fwd.ci[0] == pytest.approx(-rev.ci[1])

    @settings(max_examples=100, deadline=None)
    @given(shift=st.floats(min_value=-5, max_value=5, allow_nan=False))
    def test_adding_a_constant_to_both_arms_changes_nothing(self, shift: float):
        base = welch_test(CONTROL, TREATMENT)
        moved = welch_test(CONTROL + shift, TREATMENT + shift)
        assert base.point == pytest.approx(moved.point)
        assert base.p_value == pytest.approx(moved.p_value)

    @settings(max_examples=60, deadline=None)
    @given(n=st.integers(min_value=30, max_value=400))
    def test_ci_width_shrinks_with_n(self, n: int):
        rng = np.random.default_rng(n)
        small = welch_test(rng.normal(0, 1, n), rng.normal(0.2, 1, n))
        large = welch_test(rng.normal(0, 1, n * 8), rng.normal(0.2, 1, n * 8))
        assert large.ci[1] - large.ci[0] < small.ci[1] - small.ci[0]

    def test_point_lies_inside_interval(self):
        r = welch_test(CONTROL, TREATMENT)
        assert r.ci[0] <= r.point <= r.ci[1]


class TestEstimateWrapper:
    def test_returns_populated_estimate(self):
        exp = make_cookie_cats_like(n=10_000, seed=4)
        est = estimate_welch(exp.data, Estimand(outcome="sum_gamerounds", treatment="version"))
        assert est.method == "welch_t"
        assert est.assumptions
        assert "df" in est.diagnostics
        assert est.ci[0] <= est.point <= est.ci[1]

    def test_skew_warning_appears_for_gamerounds(self):
        """sum_gamerounds has mean/median > 2, so the estimate must say so."""
        exp = make_cookie_cats_like(n=10_000, seed=4)
        est = estimate_welch(exp.data, Estimand(outcome="sum_gamerounds", treatment="version"))
        assert any("heavily skewed" in a for a in est.assumptions)

    def test_no_skew_warning_for_a_symmetric_metric(self):
        exp = make_cookie_cats_like(n=10_000, seed=4)
        est = estimate_welch(exp.data, Estimand(outcome="retention_1", treatment="version"))
        assert not any("heavily skewed" in a for a in est.assumptions)

    def test_welch_on_binary_matches_the_proportion_difference(self):
        """A proportion is the mean of an indicator, so the point estimates agree."""
        from gatekeeper.frequentist.proportions import estimate_two_proportion

        exp = make_cookie_cats_like(n=20_000, seed=6)
        estimand = Estimand(outcome="retention_7", treatment="version")
        w = estimate_welch(exp.data, estimand)
        z = estimate_two_proportion(exp.data, estimand)
        assert w.point == pytest.approx(z.point)
        # The SEs agree closely too: Welch's unpooled variance on an indicator IS the
        # unpooled binomial variance, up to the ddof=1 correction.
        assert w.se == pytest.approx(z.se, rel=1e-3)

    def test_relative_scale(self):
        exp = make_cookie_cats_like(n=10_000, seed=4)
        est = estimate_welch(
            exp.data,
            Estimand(outcome="sum_gamerounds", treatment="version", scale=Scale.RELATIVE),
        )
        assert est.ci[0] <= est.point <= est.ci[1]
        assert any("log ratio" in a for a in est.assumptions)

    def test_relative_scale_rejects_non_positive_mean(self):
        import pandas as pd

        from gatekeeper.data.schema import ExperimentData
        from gatekeeper.types import DataSource

        df = pd.DataFrame(
            {
                "userid": range(1, 11),
                "version": ["gate_30"] * 5 + ["gate_40"] * 5,
                "sum_gamerounds": [0] * 5 + [1, 2, 3, 4, 5],
                "retention_1": [False] * 10,
                "retention_7": [False] * 10,
            }
        )
        data = ExperimentData.from_frame(df, data_source=DataSource.SYNTHETIC)
        with pytest.raises(InsufficientData, match="both arm means positive"):
            estimate_welch(
                data,
                Estimand(outcome="sum_gamerounds", treatment="version", scale=Scale.RELATIVE),
            )
