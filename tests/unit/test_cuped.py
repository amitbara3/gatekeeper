"""CUPED: the variance-reduction maths, and the guard that refuses a mediator."""

from __future__ import annotations

import numpy as np
import pytest

from gatekeeper.data.synthetic import make_cookie_cats_like, make_pre_period_experiment
from gatekeeper.frequentist.means import welch_test
from gatekeeper.types import (
    Estimand,
    InsufficientData,
    PostTreatmentCovariateError,
    Scale,
)
from gatekeeper.variance.cuped import cuped_adjust, cuped_theta, estimate_cuped


class TestPostTreatmentGuard:
    """R1.7 -- the single most emphasised rule in the project, now enforced.

    Using ``sum_gamerounds`` as the covariate would *appear* to work: it correlates
    strongly with retention, so the measured variance would drop substantially. It is
    measured after the player meets the gate, so the drop is bias, not a win.
    """

    def test_sum_gamerounds_is_refused_on_cookie_cats(self):
        exp = make_cookie_cats_like(n=2_000, seed=1)
        with pytest.raises(PostTreatmentCovariateError, match="after treatment assignment"):
            estimate_cuped(
                exp.data,
                Estimand(outcome="retention_7", treatment="version"),
                covariate="sum_gamerounds",
            )

    def test_retention_is_also_refused(self):
        exp = make_cookie_cats_like(n=2_000, seed=1)
        with pytest.raises(PostTreatmentCovariateError):
            estimate_cuped(
                exp.data,
                Estimand(outcome="sum_gamerounds", treatment="version"),
                covariate="retention_1",
            )

    def test_error_explains_the_mediator_problem(self):
        exp = make_cookie_cats_like(n=2_000, seed=1)
        with pytest.raises(PostTreatmentCovariateError, match="mediator"):
            estimate_cuped(
                exp.data,
                Estimand(outcome="retention_7", treatment="version"),
                covariate="sum_gamerounds",
            )

    def test_guard_fires_before_any_computation(self):
        """The failure must be about the analysis plan, not about arithmetic.

        A one-row-per-arm frame cannot support a t-test at all; if the guard ran after
        the maths we would see InsufficientData instead of the R1.7 error.
        """
        import pandas as pd

        from gatekeeper.data.schema import ExperimentData
        from gatekeeper.types import DataSource

        df = pd.DataFrame(
            {
                "userid": [1, 2],
                "version": ["gate_30", "gate_40"],
                "sum_gamerounds": [5, 6],
                "retention_1": [True, False],
                "retention_7": [True, False],
            }
        )
        data = ExperimentData.from_frame(df, data_source=DataSource.SYNTHETIC)
        with pytest.raises(PostTreatmentCovariateError):
            estimate_cuped(
                data,
                Estimand(outcome="retention_7", treatment="version"),
                covariate="sum_gamerounds",
            )

    def test_a_genuine_pre_period_covariate_is_accepted(self):
        exp = make_pre_period_experiment(n=2_000, seed=1)
        est = estimate_cuped(
            exp.data, Estimand(outcome="rounds", treatment="version"), covariate="pre_rounds"
        )
        assert est.method == "cuped_welch_t"


class TestTheta:
    def test_hand_computed(self):
        # y = 2x exactly -> cov(y,x)/var(x) = 2
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x
        assert cuped_theta(y, x) == pytest.approx(2.0)

    def test_theta_is_zero_for_an_uncorrelated_covariate(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 20_000)
        y = rng.normal(0, 1, 20_000)
        assert cuped_theta(y, x) == pytest.approx(0.0, abs=0.02)

    def test_zero_variance_covariate_raises(self):
        with pytest.raises(InsufficientData, match="zero variance"):
            cuped_theta(np.arange(5.0), np.ones(5))

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="align elementwise"):
            cuped_theta(np.arange(5.0), np.arange(6.0))

    def test_too_few_observations_raise(self):
        with pytest.raises(InsufficientData, match=">= 2 observations"):
            cuped_theta(np.array([1.0]), np.array([1.0]))


class TestAdjustment:
    def test_adjustment_shifts_the_estimate_only_by_the_noise_it_removes(self):
        """CUPED shrinks the interval; the estimate moves only by chance imbalance.

        The adjusted difference is *not* identical to the raw one. They differ by exactly
        ``theta * (mean(X_treatment) - mean(X_control))`` -- the covariate imbalance that
        arose by chance, which is precisely the noise CUPED exists to remove.

        So the bound here is derived rather than guessed: that shift has standard
        deviation ``|theta| * sd_x * sqrt(1/n_c + 1/n_t)``, and the test allows 4 of them.
        An earlier version used a hard-coded 0.15 and failed on an unlucky seed at 2.4
        SD, which was the tolerance being wrong rather than the code.
        """
        exp = make_pre_period_experiment(n=40_000, seed=2, rho=0.8, effect=2.0)
        data = exp.data
        y_c = data.outcome("rounds", "control")
        y_t = data.outcome("rounds", "treatment")
        x_c = data.outcome("pre_rounds", "control")
        x_t = data.outcome("pre_rounds", "treatment")

        raw = welch_test(y_c, y_t)
        adj = cuped_adjust(y_c, x_c, y_t, x_t)
        adjusted = welch_test(adj.adjusted_control, adj.adjusted_treatment)

        # The shift is exactly theta * covariate imbalance -- assert the identity.
        expected_shift = -adj.theta * (float(x_t.mean()) - float(x_c.mean()))
        assert (adjusted.point - raw.point) == pytest.approx(expected_shift, abs=1e-9)

        # And that shift is small: within 4 SD of zero.
        pooled_sd_x = float(np.concatenate([x_c, x_t]).std(ddof=1))
        shift_sd = abs(adj.theta) * pooled_sd_x * np.sqrt(1 / x_c.size + 1 / x_t.size)
        assert abs(adjusted.point - raw.point) <= 4 * shift_sd

        # Both intervals cover the truth, and CUPED's is tighter -- the whole point.
        assert raw.ci[0] <= 2.0 <= raw.ci[1]
        assert adjusted.ci[0] <= 2.0 <= adjusted.ci[1]
        assert adjusted.se < raw.se

    def test_achieved_reduction_matches_theory(self):
        """The load-bearing check: the fraction of variance removed must equal rho^2.

        Note the direction. CUPED *multiplies* the variance by ``1 - rho^2``, so the
        fraction *removed* is ``rho^2``. Confusing the two is easy -- this test caught
        exactly that inversion in ``theoretical_reduction``.
        """
        for rho in (0.3, 0.5, 0.7, 0.9):
            exp = make_pre_period_experiment(n=60_000, seed=3, rho=rho)
            data = exp.data
            adj = cuped_adjust(
                data.outcome("rounds", "control"),
                data.outcome("pre_rounds", "control"),
                data.outcome("rounds", "treatment"),
                data.outcome("pre_rounds", "treatment"),
            )
            assert adj.rho == pytest.approx(rho, abs=0.02), f"rho recovery failed at {rho}"
            assert adj.theoretical_reduction == pytest.approx(rho**2, abs=0.02)
            assert adj.achieved_reduction == pytest.approx(rho**2, abs=0.02), (
                f"at rho={rho}, achieved {adj.achieved_reduction:.4f} vs theoretical "
                f"rho^2 = {rho**2:.4f}"
            )
            # And the variance RATIO is 1 - rho^2, the other half of the same statement.
            assert (1 - adj.achieved_reduction) == pytest.approx(1 - rho**2, abs=0.02)

    def test_theta_recovers_the_generating_slope(self):
        """With pre_sd == outcome_sd, theta should be approximately rho."""
        exp = make_pre_period_experiment(n=60_000, seed=4, rho=0.6, pre_sd=8.0, outcome_sd=8.0)
        data = exp.data
        adj = cuped_adjust(
            data.outcome("rounds", "control"),
            data.outcome("pre_rounds", "control"),
            data.outcome("rounds", "treatment"),
            data.outcome("pre_rounds", "treatment"),
        )
        assert adj.theta == pytest.approx(0.6, abs=0.03)

    def test_useless_covariate_gives_no_reduction(self):
        exp = make_pre_period_experiment(n=40_000, seed=5, rho=0.0)
        data = exp.data
        adj = cuped_adjust(
            data.outcome("rounds", "control"),
            data.outcome("pre_rounds", "control"),
            data.outcome("rounds", "treatment"),
            data.outcome("pre_rounds", "treatment"),
        )
        assert adj.achieved_reduction == pytest.approx(0.0, abs=0.02)

    def test_effective_sample_size_multiplier(self):
        exp = make_pre_period_experiment(n=60_000, seed=6, rho=0.707)
        data = exp.data
        adj = cuped_adjust(
            data.outcome("rounds", "control"),
            data.outcome("pre_rounds", "control"),
            data.outcome("rounds", "treatment"),
            data.outcome("pre_rounds", "treatment"),
        )
        # rho ~ 1/sqrt(2) halves the variance, worth ~2x the sample.
        assert adj.effective_sample_size_multiplier == pytest.approx(2.0, abs=0.15)

    def test_short_arm_raises(self):
        with pytest.raises(InsufficientData, match="needs >= 2"):
            cuped_adjust(np.array([1.0]), np.array([1.0]), np.arange(5.0), np.arange(5.0))

    def test_mismatched_arm_lengths_raise(self):
        with pytest.raises(ValueError, match="lengths differ"):
            cuped_adjust(np.arange(5.0), np.arange(4.0), np.arange(5.0), np.arange(5.0))


class TestEstimateCuped:
    def test_recovers_the_known_true_effect(self):
        exp = make_pre_period_experiment(n=40_000, seed=7, rho=0.7, effect=2.0)
        est = estimate_cuped(
            exp.data, Estimand(outcome="rounds", treatment="version"), covariate="pre_rounds"
        )
        assert est.ci[0] <= exp.true_effect("rounds") <= est.ci[1]
        assert est.point == pytest.approx(2.0, abs=0.3)

    def test_interval_is_narrower_than_the_unadjusted_one(self):
        exp = make_pre_period_experiment(n=40_000, seed=8, rho=0.8, effect=2.0)
        estimand = Estimand(outcome="rounds", treatment="version")
        from gatekeeper.frequentist.means import estimate_welch

        plain = estimate_welch(exp.data, estimand)
        adjusted = estimate_cuped(exp.data, estimand, covariate="pre_rounds")
        assert adjusted.ci_width < plain.ci_width
        # rho=0.8 predicts a 36% variance reduction, so ~20% narrower interval.
        assert adjusted.ci_width / plain.ci_width == pytest.approx(0.6, abs=0.1)

    def test_diagnostics_are_populated(self):
        exp = make_pre_period_experiment(n=10_000, seed=9, rho=0.6)
        est = estimate_cuped(
            exp.data, Estimand(outcome="rounds", treatment="version"), covariate="pre_rounds"
        )
        for key in (
            "theta",
            "rho",
            "theoretical_reduction",
            "achieved_reduction",
            "effective_n_multiplier",
        ):
            assert key in est.diagnostics

    def test_assumptions_name_the_pre_treatment_requirement(self):
        exp = make_pre_period_experiment(n=5_000, seed=10)
        est = estimate_cuped(
            exp.data, Estimand(outcome="rounds", treatment="version"), covariate="pre_rounds"
        )
        assert any("BEFORE assignment" in a for a in est.assumptions)
        assert any("POOLED" in a for a in est.assumptions)

    def test_synthetic_provenance_is_recorded(self):
        exp = make_pre_period_experiment(n=5_000, seed=11)
        est = estimate_cuped(
            exp.data, Estimand(outcome="rounds", treatment="version"), covariate="pre_rounds"
        )
        assert est.is_synthetic
        assert any("synthetic" in a for a in est.assumptions)

    def test_useless_covariate_warns_on_the_estimate(self):
        exp = make_pre_period_experiment(n=20_000, seed=12, rho=0.0)
        est = estimate_cuped(
            exp.data, Estimand(outcome="rounds", treatment="version"), covariate="pre_rounds"
        )
        assert any("did not reduce variance" in a for a in est.assumptions)

    def test_relative_scale_is_refused_rather_than_faked(self):
        exp = make_pre_period_experiment(n=2_000, seed=13)
        with pytest.raises(NotImplementedError, match="absolute scale only"):
            estimate_cuped(
                exp.data,
                Estimand(outcome="rounds", treatment="version", scale=Scale.RELATIVE),
                covariate="pre_rounds",
            )


class TestUnbiasedness:
    """The claim that makes CUPED safe: it reduces variance without adding bias."""

    def test_cuped_is_unbiased_across_many_experiments(self):
        true_effect = 2.0
        plain_estimates, cuped_estimates = [], []
        for seed in range(120):
            exp = make_pre_period_experiment(n=3_000, seed=seed, rho=0.7, effect=true_effect)
            data = exp.data
            y_c = data.outcome("rounds", "control")
            y_t = data.outcome("rounds", "treatment")
            x_c = data.outcome("pre_rounds", "control")
            x_t = data.outcome("pre_rounds", "treatment")
            plain_estimates.append(welch_test(y_c, y_t).point)
            adj = cuped_adjust(y_c, x_c, y_t, x_t)
            cuped_estimates.append(welch_test(adj.adjusted_control, adj.adjusted_treatment).point)

        plain = np.array(plain_estimates)
        cuped = np.array(cuped_estimates)

        # Both unbiased...
        assert plain.mean() == pytest.approx(true_effect, abs=0.1)
        assert cuped.mean() == pytest.approx(true_effect, abs=0.1)
        # ...but CUPED is markedly less variable. rho=0.7 predicts variance x0.51,
        # i.e. sd x ~0.71.
        assert cuped.std(ddof=1) < plain.std(ddof=1)
        assert cuped.std(ddof=1) / plain.std(ddof=1) == pytest.approx(0.714, abs=0.1)
