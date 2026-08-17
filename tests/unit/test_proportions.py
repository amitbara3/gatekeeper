"""Two-proportion tests, with hand-computed known answers."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gatekeeper.data.synthetic import make_cookie_cats_like
from gatekeeper.frequentist.proportions import estimate_two_proportion, two_proportion_test
from gatekeeper.types import DataSource, Estimand, InsufficientData, Scale


class TestKnownAnswers:
    def test_identical_arms_give_zero_effect_and_p_one(self):
        r = two_proportion_test(200, 1000, 200, 1000)
        assert r.point == pytest.approx(0.0)
        assert r.p_value == pytest.approx(1.0)
        assert r.z == pytest.approx(0.0)

    def test_hand_computed_case(self):
        # p_c = 0.20, p_t = 0.25, diff = 0.05
        # pooled p = 450/2000 = 0.225
        # se_pooled = sqrt(0.225*0.775*(1/1000+1/1000)) = sqrt(0.174375*0.002)
        r = two_proportion_test(200, 1000, 250, 1000)
        assert r.p_control == pytest.approx(0.20)
        assert r.p_treatment == pytest.approx(0.25)
        assert r.point == pytest.approx(0.05)

        expected_se_pooled = math.sqrt(0.225 * 0.775 * (1 / 1000 + 1 / 1000))
        assert r.se_pooled == pytest.approx(expected_se_pooled)
        assert r.z == pytest.approx(0.05 / expected_se_pooled)

        # CI uses the UNPOOLED se
        expected_se_unpooled = math.sqrt(0.2 * 0.8 / 1000 + 0.25 * 0.75 / 1000)
        assert r.se == pytest.approx(expected_se_unpooled)
        lo, hi = r.ci
        assert lo == pytest.approx(0.05 - 1.959963985 * expected_se_unpooled, rel=1e-6)
        assert hi == pytest.approx(0.05 + 1.959963985 * expected_se_unpooled, rel=1e-6)

    def test_pooled_and_unpooled_se_genuinely_differ(self):
        """If these ever coincide the split has been silently collapsed."""
        r = two_proportion_test(100, 1000, 400, 1000)
        assert r.se_pooled != pytest.approx(r.se, rel=1e-9)

    def test_relative_scale_point_estimate(self):
        # p_c = 0.20 -> p_t = 0.25 is a +25% relative change
        r = two_proportion_test(200, 1000, 250, 1000, scale=Scale.RELATIVE)
        assert r.point == pytest.approx(0.25)

    def test_relative_ci_cannot_go_below_minus_one(self):
        """The log scale makes a sub -100% relative change impossible by construction."""
        r = two_proportion_test(500, 1000, 5, 1000, scale=Scale.RELATIVE)
        assert r.ci[0] > -1.0
        # p_c = 0.5, p_t = 0.005 -> ratio 0.01, i.e. a 99% relative decrease.
        assert r.point == pytest.approx(0.005 / 0.5 - 1.0)

    def test_p_value_is_identical_across_scales(self):
        """The null (equal rates) is the same hypothesis on either scale."""
        abs_r = two_proportion_test(200, 1000, 250, 1000, scale=Scale.ABSOLUTE)
        rel_r = two_proportion_test(200, 1000, 250, 1000, scale=Scale.RELATIVE)
        assert abs_r.p_value == pytest.approx(rel_r.p_value)

    def test_zero_treatment_rate_is_exactly_minus_one_hundred_percent(self):
        r = two_proportion_test(200, 1000, 0, 1000, scale=Scale.RELATIVE)
        assert r.point == pytest.approx(-1.0)
        assert r.ci == (-1.0, -1.0)

    def test_all_success_both_arms_gives_no_evidence(self):
        r = two_proportion_test(1000, 1000, 1000, 1000)
        assert r.point == pytest.approx(0.0)
        assert r.p_value == pytest.approx(1.0)
        assert r.se_pooled == 0.0


class TestValidation:
    def test_empty_arm_raises(self):
        with pytest.raises(InsufficientData, match="at least one unit"):
            two_proportion_test(0, 0, 5, 10)

    def test_successes_above_n_raises(self):
        with pytest.raises(ValueError, match=r"must be in \[0, n\]"):
            two_proportion_test(20, 10, 5, 10)

    def test_negative_successes_raises(self):
        with pytest.raises(ValueError, match=r"must be in \[0, n\]"):
            two_proportion_test(-1, 10, 5, 10)

    def test_bad_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            two_proportion_test(5, 10, 6, 10, alpha=0.0)

    def test_relative_with_zero_control_raises(self):
        with pytest.raises(InsufficientData, match="control rate is zero"):
            two_proportion_test(0, 1000, 50, 1000, scale=Scale.RELATIVE)

    def test_thin_arm_warns_about_normal_approximation(self):
        with pytest.warns(UserWarning, match="normal approximation is unreliable"):
            two_proportion_test(2, 100, 3, 100)

    def test_warning_can_be_suppressed_for_simulation(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            two_proportion_test(2, 100, 3, 100, warn_small=False)


class TestProperties:
    @settings(max_examples=200, deadline=None)
    @given(
        s_c=st.integers(min_value=1, max_value=999),
        s_t=st.integers(min_value=1, max_value=999),
    )
    def test_swapping_arms_flips_sign_and_keeps_p_value(self, s_c: int, s_t: int):
        forward = two_proportion_test(s_c, 1000, s_t, 1000, warn_small=False)
        reverse = two_proportion_test(s_t, 1000, s_c, 1000, warn_small=False)
        assert forward.point == pytest.approx(-reverse.point)
        assert forward.p_value == pytest.approx(reverse.p_value)
        assert forward.ci[0] == pytest.approx(-reverse.ci[1])
        assert forward.ci[1] == pytest.approx(-reverse.ci[0])

    @settings(max_examples=100, deadline=None)
    @given(n=st.integers(min_value=200, max_value=200_000))
    def test_ci_width_shrinks_as_n_grows(self, n: int):
        small = two_proportion_test(n // 5, n, n // 4, n, warn_small=False)
        large = two_proportion_test(n, 5 * n, (5 * n) // 4, 5 * n, warn_small=False)
        assert large.ci[1] - large.ci[0] < small.ci[1] - small.ci[0]

    @settings(max_examples=200, deadline=None)
    @given(
        s_c=st.integers(min_value=1, max_value=999),
        s_t=st.integers(min_value=1, max_value=999),
    )
    def test_point_estimate_lies_inside_its_own_interval(self, s_c: int, s_t: int):
        r = two_proportion_test(s_c, 1000, s_t, 1000, warn_small=False)
        assert r.ci[0] <= r.point <= r.ci[1]

    @settings(max_examples=100, deadline=None)
    @given(
        s_c=st.integers(min_value=10, max_value=990),
        s_t=st.integers(min_value=10, max_value=990),
    )
    def test_wider_alpha_gives_narrower_interval(self, s_c: int, s_t: int):
        tight = two_proportion_test(s_c, 1000, s_t, 1000, alpha=0.01, warn_small=False)
        loose = two_proportion_test(s_c, 1000, s_t, 1000, alpha=0.10, warn_small=False)
        assert loose.ci[1] - loose.ci[0] < tight.ci[1] - tight.ci[0]


class TestEstimateWrapper:
    def test_returns_populated_effect_estimate(self):
        exp = make_cookie_cats_like(n=20_000, seed=3)
        estimand = Estimand(outcome="retention_7", treatment="version")
        est = estimate_two_proportion(exp.data, estimand)

        assert est.method == "two_proportion_z"
        assert est.assumptions  # required by R2.4
        assert est.data_source is DataSource.SYNTHETIC
        assert est.ci_level == pytest.approx(0.95)
        assert set(est.n_per_arm) == {"gate_30", "gate_40"}
        assert est.ci[0] <= est.point <= est.ci[1]

    def test_recovers_the_known_true_effect(self):
        exp = make_cookie_cats_like(n=200_000, seed=5, retention_7_effect=-0.02)
        estimand = Estimand(outcome="retention_7", treatment="version")
        est = estimate_two_proportion(exp.data, estimand)
        assert est.ci[0] <= exp.true_effect("retention_7") <= est.ci[1]
        assert est.point == pytest.approx(-0.02, abs=0.005)

    def test_synthetic_data_is_flagged_in_assumptions(self):
        exp = make_cookie_cats_like(n=5_000, seed=3)
        est = estimate_two_proportion(
            exp.data, Estimand(outcome="retention_1", treatment="version")
        )
        assert any("synthetic" in a for a in est.assumptions)
        assert not any("randomised, licensing" in a for a in est.assumptions)

    def test_non_binary_metric_is_rejected(self):
        exp = make_cookie_cats_like(n=2_000, seed=3)
        with pytest.raises(ValueError, match="not binary"):
            estimate_two_proportion(
                exp.data, Estimand(outcome="sum_gamerounds", treatment="version")
            )

    def test_diagnostics_carry_both_standard_errors(self):
        exp = make_cookie_cats_like(n=20_000, seed=3)
        est = estimate_two_proportion(
            exp.data, Estimand(outcome="retention_7", treatment="version")
        )
        assert "se_pooled" in est.diagnostics
        assert "se_unpooled" in est.diagnostics
        assert est.diagnostics["se_pooled"] != pytest.approx(
            est.diagnostics["se_unpooled"], rel=1e-12
        )
