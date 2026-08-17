"""CATE learners and uplift curves, scored against a known effect function."""

from __future__ import annotations

import numpy as np
import pytest

from gatekeeper.causal.confounding import make_heterogeneous, make_randomised
from gatekeeper.hte.learners import estimate_cate
from gatekeeper.hte.uplift import qini_curve, uplift_curve
from gatekeeper.types import InsufficientData, PostTreatmentCovariateError

LEARNERS = ("s", "t", "x")


class TestRecoversKnownCate:
    """tau(x) = 1.0 + 1.5x, so both the level and the shape are checkable."""

    @pytest.mark.parametrize("learner", LEARNERS)
    def test_ate_is_recovered(self, learner: str):
        data, true_tau = make_heterogeneous(20_000, base_effect=1.0, effect_slope=1.5, seed=0)
        est = estimate_cate(data, "y", ["x"], learner=learner, treatment_column="received")
        assert est.ate == pytest.approx(float(true_tau.mean()), abs=0.2), est.describe()

    @pytest.mark.parametrize("learner", ("t", "x"))
    def test_per_unit_effects_correlate_with_the_truth(self, learner: str):
        """The T- and X-learners should recover the SHAPE, not just the average."""
        data, true_tau = make_heterogeneous(20_000, effect_slope=1.5, seed=1)
        est = estimate_cate(data, "y", ["x"], learner=learner, treatment_column="received")
        correlation = float(np.corrcoef(est.tau, true_tau)[0, 1])
        assert correlation > 0.7, f"{learner}-learner correlation only {correlation:.3f}"

    @pytest.mark.parametrize("learner", ("t", "x"))
    def test_finds_heterogeneity_when_it_exists(self, learner: str):
        data, _ = make_heterogeneous(20_000, effect_slope=1.5, seed=2)
        est = estimate_cate(data, "y", ["x"], learner=learner, treatment_column="received")
        assert est.spread > 0.5, est.describe()

    @pytest.mark.parametrize("learner", LEARNERS)
    def test_finds_no_heterogeneity_when_there_is_none(self, learner: str):
        """A homogeneous effect must not produce phantom heterogeneity."""
        data, _ = make_heterogeneous(20_000, base_effect=1.0, effect_slope=0.0, seed=3)
        est = estimate_cate(
            data, "y", ["x"], learner=learner, treatment_column="received", flexible=False
        )
        assert est.spread < 0.2, est.describe()

    @pytest.mark.parametrize("learner", ("t", "x"))
    def test_decile_effects_are_ordered(self, learner: str):
        data, _ = make_heterogeneous(20_000, effect_slope=1.5, seed=4)
        est = estimate_cate(data, "y", ["x"], learner=learner, treatment_column="received")
        deciles = est.decile_effects()
        assert deciles[-1] > deciles[0]
        # Deciles are formed by sorting tau, so they must be monotone by construction.
        assert list(deciles) == sorted(deciles)


class TestSLearnerBias:
    """The S-learner's shrinkage toward zero is a documented property, shown not asserted.

    A flexible learner given one treatment column among several covariates will often
    under-use it, because dropping a weak feature costs little prediction loss. The
    S-learner is included in the library precisely so this is visible.
    """

    def test_s_learner_finds_less_heterogeneity_than_t_and_x(self):
        data, _ = make_heterogeneous(20_000, base_effect=1.0, effect_slope=2.0, seed=5)
        spreads = {
            learner: estimate_cate(
                data, "y", ["x"], learner=learner, treatment_column="received"
            ).spread
            for learner in LEARNERS
        }
        assert spreads["s"] < spreads["t"], spreads
        assert spreads["s"] < spreads["x"], spreads


class TestValidation:
    def test_post_treatment_covariate_is_refused(self):
        data, _ = make_heterogeneous(2_000, seed=0)
        with pytest.raises(PostTreatmentCovariateError):
            estimate_cate(data, "y", ["received"], treatment_column="received")

    def test_no_covariates_raises(self):
        data, _ = make_heterogeneous(2_000, seed=0)
        with pytest.raises(ValueError, match="at least one covariate"):
            estimate_cate(data, "y", [], treatment_column="received")

    def test_unknown_learner_raises(self):
        data, _ = make_heterogeneous(2_000, seed=0)
        with pytest.raises(ValueError, match="unknown learner"):
            estimate_cate(data, "y", ["x"], learner="q", treatment_column="received")  # type: ignore[arg-type]

    def test_tiny_arm_raises(self):
        data, _ = make_heterogeneous(200, treatment_share=0.02, seed=0)
        with pytest.raises(InsufficientData, match=">= 20 units"):
            estimate_cate(data, "y", ["x"], treatment_column="received")

    def test_generator_validation(self):
        with pytest.raises(InsufficientData, match=">= 100 units"):
            make_heterogeneous(10)
        with pytest.raises(ValueError, match="treatment_share"):
            make_heterogeneous(1_000, treatment_share=1.5)


class TestUpliftCurves:
    def test_a_perfect_ranking_beats_random(self):
        data, true_tau = make_heterogeneous(20_000, effect_slope=2.0, seed=6)
        treated = data.frame["received"].to_numpy(dtype=float)
        y = data.frame["y"].to_numpy(dtype=float)
        # Rank by the TRUE effect: the best possible ordering.
        curve = qini_curve(true_tau, treated, y)
        assert curve.beats_random, curve.describe()
        assert curve.coefficient > 0.1

    def test_a_random_ranking_does_not_beat_random(self):
        data, _ = make_heterogeneous(20_000, effect_slope=2.0, seed=7)
        treated = data.frame["received"].to_numpy(dtype=float)
        y = data.frame["y"].to_numpy(dtype=float)
        noise = np.random.default_rng(0).standard_normal(y.size)
        curve = qini_curve(noise, treated, y)
        assert abs(curve.coefficient) < 0.1, curve.describe()

    def test_a_reversed_ranking_is_worse_than_random(self):
        """Negative coefficients are a real finding: the model ranks backwards."""
        data, true_tau = make_heterogeneous(20_000, effect_slope=2.0, seed=8)
        treated = data.frame["received"].to_numpy(dtype=float)
        y = data.frame["y"].to_numpy(dtype=float)
        curve = qini_curve(-true_tau, treated, y)
        assert not curve.beats_random
        assert curve.coefficient < -0.1, curve.describe()

    def test_a_learned_ranking_beats_random(self):
        data, _ = make_heterogeneous(20_000, effect_slope=2.0, seed=9)
        treated = data.frame["received"].to_numpy(dtype=float)
        y = data.frame["y"].to_numpy(dtype=float)
        est = estimate_cate(data, "y", ["x"], learner="x", treatment_column="received")
        curve = qini_curve(est.tau, treated, y)
        assert curve.beats_random, curve.describe()

    def test_no_heterogeneity_gives_no_targeting_advantage(self):
        data, true_tau = make_heterogeneous(20_000, effect_slope=0.0, seed=10)
        treated = data.frame["received"].to_numpy(dtype=float)
        y = data.frame["y"].to_numpy(dtype=float)
        curve = qini_curve(true_tau, treated, y)
        # A constant score cannot rank, so targeting cannot help.
        assert abs(curve.coefficient) < 0.15, curve.describe()

    def test_total_gain_reflects_the_overall_effect(self):
        data, true_tau = make_heterogeneous(10_000, base_effect=1.0, effect_slope=0.0, seed=11)
        treated = data.frame["received"].to_numpy(dtype=float)
        y = data.frame["y"].to_numpy(dtype=float)
        curve = uplift_curve(true_tau, treated, y)
        # At full depth the uplift curve's gain is (mean_t - mean_c) * n.
        expected = (y[treated == 1].mean() - y[treated == 0].mean()) * y.size
        assert curve.total_gain == pytest.approx(expected, rel=1e-6)

    def test_uplift_and_qini_agree_on_direction(self):
        data, true_tau = make_heterogeneous(20_000, effect_slope=2.0, seed=12)
        treated = data.frame["received"].to_numpy(dtype=float)
        y = data.frame["y"].to_numpy(dtype=float)
        assert (
            uplift_curve(true_tau, treated, y).beats_random
            == qini_curve(true_tau, treated, y).beats_random
        )

    def test_depths_span_to_one(self):
        data, true_tau = make_heterogeneous(5_000, seed=13)
        treated = data.frame["received"].to_numpy(dtype=float)
        y = data.frame["y"].to_numpy(dtype=float)
        curve = qini_curve(true_tau, treated, y, n_bins=10)
        assert curve.depths[-1] == pytest.approx(1.0)
        assert curve.depths.size == 10
        assert list(curve.depths) == sorted(curve.depths)

    def test_validation(self):
        y = np.arange(100.0)
        treated = np.tile([0.0, 1.0], 50)
        with pytest.raises(ValueError, match="must align"):
            qini_curve(np.arange(50.0), treated, y)
        with pytest.raises(ValueError, match="0/1 indicator"):
            qini_curve(y, np.arange(100.0), y)
        with pytest.raises(InsufficientData, match="at least"):
            qini_curve(y[:5], treated[:5], y[:5], n_bins=20)

    def test_describe_is_readable(self):
        data, true_tau = make_heterogeneous(5_000, seed=14)
        treated = data.frame["received"].to_numpy(dtype=float)
        y = data.frame["y"].to_numpy(dtype=float)
        assert "coefficient" in qini_curve(true_tau, treated, y).describe()


class TestCookieCatsLimitation:
    """Phase 7's honest finding: this dataset cannot support CATE estimation."""

    def test_cookie_cats_has_no_pre_treatment_covariates(self):
        from gatekeeper.data.schema import COOKIE_CATS

        usable = [
            c.name
            for c in COOKIE_CATS.columns
            if not c.post_treatment
            and c.name not in (COOKIE_CATS.unit_col, COOKIE_CATS.variant_col)
        ]
        assert usable == [], (
            f"expected no usable covariates, found {usable}. If Cookie Cats gains a "
            "pre-treatment covariate, Phase 7's conclusion needs revisiting."
        )

    def test_every_metric_is_post_treatment(self):
        from gatekeeper.data.schema import COOKIE_CATS

        for metric in COOKIE_CATS.metric_columns:
            assert metric in COOKIE_CATS.post_treatment_columns

    def test_randomised_scenario_still_supports_cate(self):
        """The machinery works; it is the dataset that cannot feed it."""
        scenario = make_randomised(5_000, seed=0)
        est = estimate_cate(scenario.data, "y", ["x"], learner="x", treatment_column="received")
        assert est.tau.size == len(scenario.data.frame)
