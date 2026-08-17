"""SRM tests, including hand-computed known answers."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gatekeeper.data.synthetic import make_cookie_cats_like
from gatekeeper.design.srm import DEFAULT_SRM_THRESHOLD, check_srm, srm_test
from gatekeeper.types import InsufficientData


class TestKnownAnswers:
    """Cases where the chi-square statistic is computable by hand."""

    def test_perfectly_balanced_split_is_exactly_zero(self):
        chi2, p = srm_test({"a": 500, "b": 500})
        assert chi2 == pytest.approx(0.0, abs=1e-12)
        assert p == pytest.approx(1.0)

    def test_600_400_split(self):
        # expected 500 each: (600-500)^2/500 + (400-500)^2/500 = 20 + 20 = 40
        chi2, p = srm_test({"a": 600, "b": 400})
        assert chi2 == pytest.approx(40.0)
        assert p < 1e-8

    def test_unequal_intended_split_that_is_met_exactly(self):
        chi2, p = srm_test({"a": 700, "b": 300}, {"a": 0.7, "b": 0.3})
        assert chi2 == pytest.approx(0.0, abs=1e-12)
        assert p == pytest.approx(1.0)

    def test_same_counts_judged_against_wrong_intended_split_fails(self):
        # 700/300 observed but 50/50 intended:
        # (700-500)^2/500 * 2 = 80 + 80 = 160
        chi2, p = srm_test({"a": 700, "b": 300}, {"a": 0.5, "b": 0.5})
        assert chi2 == pytest.approx(160.0)
        assert p < 1e-30

    def test_three_arms_uses_two_degrees_of_freedom(self):
        # 120/90/90 vs expected 100 each: 4 + 1 + 1 = 6, on df=2
        from scipy import stats

        chi2, p = srm_test({"a": 120, "b": 90, "c": 90})
        assert chi2 == pytest.approx(6.0)
        assert p == pytest.approx(float(stats.chi2.sf(6.0, df=2)))


class TestValidation:
    def test_single_arm_raises(self):
        with pytest.raises(InsufficientData, match="at least two arms"):
            srm_test({"a": 100})

    def test_all_zero_counts_raises(self):
        with pytest.raises(InsufficientData, match="at least one unit"):
            srm_test({"a": 0, "b": 0})

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            srm_test({"a": -1, "b": 10})

    def test_shares_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            srm_test({"a": 10, "b": 10}, {"a": 0.5, "b": 0.4})

    def test_share_keys_must_match_observed_arms(self):
        with pytest.raises(ValueError, match="do not match"):
            srm_test({"a": 10, "b": 10}, {"a": 0.5, "c": 0.5})

    def test_non_positive_share_raises(self):
        with pytest.raises(ValueError, match="positive"):
            srm_test({"a": 10, "b": 10}, {"a": 1.0, "b": 0.0})

    def test_check_srm_rejects_out_of_range_threshold(self, data):
        with pytest.raises(ValueError, match="must be in"):
            check_srm(data, threshold=1.5)


class TestCheckSrm:
    def test_balanced_synthetic_experiment_passes(self):
        exp = make_cookie_cats_like(n=20_000, seed=1, treatment_share=0.5)
        check = check_srm(exp.data)
        assert check.passed
        assert check.name == "srm"
        assert check.p_value is not None and check.p_value > DEFAULT_SRM_THRESHOLD

    def test_deliberately_imbalanced_experiment_fails(self):
        # A 45/55 split over 20k units is a gross mismatch against an intended 50/50.
        exp = make_cookie_cats_like(n=20_000, seed=1, treatment_share=0.45)
        check = check_srm(exp.data)
        assert not check.passed
        assert "SAMPLE RATIO MISMATCH" in check.detail
        assert check.threshold == DEFAULT_SRM_THRESHOLD

    def test_detail_reports_observed_shares(self, data):
        check = check_srm(data)
        assert "gate_30=" in check.detail
        assert "gate_40=" in check.detail

    def test_threshold_is_recorded_for_reproducibility(self, data):
        check = check_srm(data, threshold=0.01)
        assert check.threshold == 0.01


class TestProperties:
    @settings(max_examples=200, deadline=None)
    @given(
        a=st.integers(min_value=1, max_value=10**6),
        b=st.integers(min_value=1, max_value=10**6),
    )
    def test_p_value_always_in_unit_interval(self, a: int, b: int):
        _, p = srm_test({"a": a, "b": b})
        assert 0.0 <= p <= 1.0

    @settings(max_examples=200, deadline=None)
    @given(n=st.integers(min_value=1, max_value=10**6))
    def test_equal_counts_always_give_zero_statistic(self, n: int):
        chi2, p = srm_test({"a": n, "b": n})
        assert chi2 == pytest.approx(0.0, abs=1e-9)
        assert p == pytest.approx(1.0)

    @settings(max_examples=100, deadline=None)
    @given(
        a=st.integers(min_value=1, max_value=10**5),
        b=st.integers(min_value=1, max_value=10**5),
    )
    def test_statistic_is_invariant_to_arm_labelling_order(self, a: int, b: int):
        """Relabelling the arms must not change the goodness-of-fit statistic."""
        chi2_1, _ = srm_test({"alpha": a, "beta": b})
        chi2_2, _ = srm_test({"zeta": b, "eta": a})
        assert chi2_1 == pytest.approx(chi2_2)

    @settings(max_examples=100, deadline=None)
    @given(
        total=st.integers(min_value=100, max_value=10**5),
        excess=st.integers(min_value=1, max_value=50),
    )
    def test_statistic_grows_with_imbalance(self, total: int, excess: int):
        """More imbalance at fixed n cannot reduce the statistic."""
        half = total // 2
        near, far = {"a": half, "b": half}, {"a": half + excess, "b": half - excess}
        if far["b"] <= 0:
            return
        chi2_near, _ = srm_test(near)
        chi2_far, _ = srm_test(far)
        assert chi2_far >= chi2_near - 1e-9


class TestFalsePositiveRate:
    """The threshold's whole justification is its false-positive rate."""

    def test_healthy_experiments_almost_never_flag_at_the_default_threshold(self):
        rng = np.random.default_rng(20260817)
        n_sims, n_units, flagged = 400, 20_000, 0
        for _ in range(n_sims):
            # A genuinely fair coin per unit -- the null the check is calibrated for.
            treated = int(rng.binomial(n_units, 0.5))
            _, p = srm_test({"control": n_units - treated, "treatment": treated})
            flagged += p < DEFAULT_SRM_THRESHOLD
        # Expected ~0.2 of 400 at p<0.0005. Allow generous headroom; the point is
        # that a strict threshold does not cry wolf on healthy experiments.
        assert flagged <= 3, f"{flagged}/{n_sims} healthy experiments flagged as SRM"
