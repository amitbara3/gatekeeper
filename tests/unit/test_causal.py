"""Causal estimators, the confounding simulator, and sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pytest

from gatekeeper.causal.aipw import estimate_aipw, estimate_outcome_regression
from gatekeeper.causal.confounding import (
    covariate_imbalance,
    make_confounded,
    make_randomised,
)
from gatekeeper.causal.propensity import (
    estimate_ipw,
    estimate_naive_difference,
    fit_propensity,
)
from gatekeeper.causal.sensitivity import e_value, e_value_from_difference
from gatekeeper.types import (
    DataSource,
    Estimand,
    EstimandTarget,
    InsufficientData,
    PostTreatmentCovariateError,
)

ESTIMAND = Estimand(outcome="y", treatment="received")


class TestConfoundingSimulator:
    def test_randomised_sample_is_balanced(self):
        s = make_randomised(20_000, seed=0)
        assert abs(covariate_imbalance(s, "x")) < 0.05
        assert abs(covariate_imbalance(s, "u")) < 0.05
        assert s.regime is None
        assert s.data.data_source is DataSource.SYNTHETIC

    def test_selection_unbalances_the_observed_covariate(self):
        s = make_confounded("selection", 20_000, strength=1.2, seed=0)
        assert abs(covariate_imbalance(s, "x")) > 0.5
        # It should NOT disturb u, which is the contrast that makes the regime specific.
        assert abs(covariate_imbalance(s, "u")) < 0.1

    def test_unobserved_unbalances_only_the_hidden_confounder(self):
        s = make_confounded("unobserved", 20_000, strength=1.2, seed=0)
        assert abs(covariate_imbalance(s, "u")) > 0.5
        assert abs(covariate_imbalance(s, "x")) < 0.1

    def test_semi_synthetic_provenance(self):
        s = make_confounded("selection", 2_000, seed=0)
        assert s.data.data_source is DataSource.SEMI_SYNTHETIC

    def test_noncompliance_breaks_the_assigned_received_link(self):
        s = make_confounded("noncompliance", 20_000, strength=1.2, seed=0)
        frame = s.data.frame
        assigned = (frame["assigned"] == "treatment").to_numpy()
        received = frame["received"].to_numpy() == 1.0
        # One-sided: nobody receives without being assigned...
        assert not (received & ~assigned).any()
        # ...but some assigned units do not comply.
        assert (assigned & ~received).sum() > 0

    def test_zero_strength_reduces_to_randomised(self):
        """Continuity check: the regimes must not do anything at strength 0."""
        for regime in ("selection", "unobserved"):
            s = make_confounded(regime, 20_000, strength=0.0, seed=1)
            assert abs(covariate_imbalance(s, "x")) < 0.05
            assert abs(covariate_imbalance(s, "u")) < 0.05

    def test_stronger_confounding_gives_more_imbalance(self):
        weak = abs(covariate_imbalance(make_confounded("selection", 20_000, strength=0.3, seed=2)))
        strong = abs(
            covariate_imbalance(make_confounded("selection", 20_000, strength=1.5, seed=2))
        )
        assert strong > weak

    def test_selection_discards_units(self):
        s = make_confounded("selection", 20_000, strength=1.2, seed=0)
        assert s.n_retained < s.n_generated
        assert 0.3 < s.retention_rate < 0.7

    def test_true_ate_is_recorded_exactly(self):
        s = make_confounded("selection", 2_000, true_ate=0.75, seed=0)
        assert s.true_ate == 0.75

    def test_validation(self):
        with pytest.raises(InsufficientData, match=">= 100 units"):
            make_randomised(10)
        with pytest.raises(ValueError, match="unknown regime"):
            make_confounded("magic", 2_000)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="strength must be non-negative"):
            make_confounded("selection", 2_000, strength=-1.0)

    def test_describe_is_readable(self):
        text = make_confounded("selection", 2_000, seed=0).describe()
        assert "selection" in text
        assert "true ATE" in text


class TestPropensity:
    def test_fit_recovers_balanced_scores_under_randomisation(self):
        s = make_randomised(20_000, seed=0)
        fit = fit_propensity(s.data, ["x"], treatment_column="received")
        # Randomised: propensity should sit near 0.5 for everyone.
        assert fit.min_score > 0.4
        assert fit.max_score < 0.6
        assert fit.has_good_overlap

    def test_selection_produces_spread_out_scores(self):
        s = make_confounded("selection", 20_000, strength=1.2, seed=0)
        fit = fit_propensity(s.data, ["x"], treatment_column="received")
        assert fit.max_score - fit.min_score > 0.5

    def test_effective_sample_size_falls_with_worse_overlap(self):
        good = fit_propensity(
            make_randomised(20_000, seed=0).data, ["x"], treatment_column="received"
        )
        poor = fit_propensity(
            make_confounded("selection", 20_000, strength=2.5, seed=0).data,
            ["x"],
            treatment_column="received",
        )
        assert poor.ess_fraction < good.ess_fraction

    def test_post_treatment_covariate_is_refused(self):
        s = make_randomised(2_000, seed=0)
        with pytest.raises(PostTreatmentCovariateError):
            fit_propensity(s.data, ["y"], treatment_column="received")

    def test_no_covariates_raises(self):
        s = make_randomised(2_000, seed=0)
        with pytest.raises(ValueError, match="at least one covariate"):
            fit_propensity(s.data, [], treatment_column="received")

    def test_describe_reports_overlap(self):
        fit = fit_propensity(
            make_randomised(2_000, seed=0).data, ["x"], treatment_column="received"
        )
        assert "propensity in" in fit.describe()
        assert "ESS" in fit.describe()


class TestEstimatorsOnRandomisedData:
    """Every estimator must recover the truth when there is no confounding.

    One that fails here is broken, not challenged -- which is why the benchmark includes
    a randomised control condition.
    """

    @pytest.mark.parametrize(
        "estimate",
        [
            lambda d: estimate_naive_difference(d, ESTIMAND, treatment_column="received"),
            lambda d: estimate_ipw(d, ESTIMAND, ["x"], treatment_column="received"),
            lambda d: estimate_outcome_regression(d, ESTIMAND, ["x"], treatment_column="received"),
            lambda d: estimate_aipw(d, ESTIMAND, ["x"], treatment_column="received"),
        ],
    )
    def test_recovers_the_true_ate(self, estimate):
        s = make_randomised(20_000, true_ate=1.0, seed=0)
        est = estimate(s.data)
        assert est.point == pytest.approx(1.0, abs=0.1)
        assert est.ci[0] <= 1.0 <= est.ci[1]

    def test_all_estimators_declare_assumptions(self):
        s = make_randomised(5_000, seed=0)
        for est in (
            estimate_naive_difference(s.data, ESTIMAND, treatment_column="received"),
            estimate_ipw(s.data, ESTIMAND, ["x"], treatment_column="received"),
            estimate_outcome_regression(s.data, ESTIMAND, ["x"], treatment_column="received"),
            estimate_aipw(s.data, ESTIMAND, ["x"], treatment_column="received"),
        ):
            assert est.assumptions


class TestEstimatorsUnderSelection:
    def test_naive_is_badly_biased(self):
        s = make_confounded("selection", 20_000, true_ate=1.0, strength=1.2, seed=0)
        est = estimate_naive_difference(s.data, ESTIMAND, treatment_column="received")
        assert abs(est.point - 1.0) > 1.0
        assert not (est.ci[0] <= 1.0 <= est.ci[1])

    @pytest.mark.parametrize(
        "estimate",
        [
            lambda d: estimate_ipw(d, ESTIMAND, ["x"], treatment_column="received"),
            lambda d: estimate_outcome_regression(d, ESTIMAND, ["x"], treatment_column="received"),
            lambda d: estimate_aipw(d, ESTIMAND, ["x"], treatment_column="received"),
        ],
    )
    def test_adjustment_recovers_the_truth(self, estimate):
        s = make_confounded("selection", 20_000, true_ate=1.0, strength=1.2, seed=0)
        est = estimate(s.data)
        assert est.point == pytest.approx(1.0, abs=0.15)

    def test_naive_assumptions_warn_it_is_only_valid_under_randomisation(self):
        s = make_confounded("selection", 2_000, seed=0)
        est = estimate_naive_difference(s.data, ESTIMAND, treatment_column="received")
        assert any("ONLY under randomised" in a for a in est.assumptions)


class TestUnobservedConfoundingDefeatsEverything:
    """The finding that justifies the whole exercise.

    Every adjustment method fails when the confounder is withheld -- and the same methods
    succeed the moment it is supplied. That contrast is what turns "AIPW is doubly robust"
    from a slogan into a precise, bounded claim: robust to MISSPECIFICATION, not to
    OMISSION.
    """

    @pytest.mark.parametrize("method", ["ipw", "outcome_regression", "aipw"])
    def test_every_method_fails_without_the_confounder(self, method: str):
        s = make_confounded("unobserved", 20_000, true_ate=1.0, strength=1.2, seed=0)
        fns = {
            "ipw": lambda: estimate_ipw(s.data, ESTIMAND, ["x"], treatment_column="received"),
            "outcome_regression": lambda: estimate_outcome_regression(
                s.data, ESTIMAND, ["x"], treatment_column="received"
            ),
            "aipw": lambda: estimate_aipw(s.data, ESTIMAND, ["x"], treatment_column="received"),
        }
        est = fns[method]()
        assert abs(est.point - 1.0) > 1.0, f"{method} unexpectedly survived"
        assert not (est.ci[0] <= 1.0 <= est.ci[1])

    @pytest.mark.parametrize("method", ["ipw", "outcome_regression", "aipw"])
    def test_every_method_succeeds_once_the_confounder_is_supplied(self, method: str):
        s = make_confounded("unobserved", 20_000, true_ate=1.0, strength=1.2, seed=0)
        fns = {
            "ipw": lambda: estimate_ipw(s.data, ESTIMAND, ["x", "u"], treatment_column="received"),
            "outcome_regression": lambda: estimate_outcome_regression(
                s.data, ESTIMAND, ["x", "u"], treatment_column="received"
            ),
            "aipw": lambda: estimate_aipw(
                s.data, ESTIMAND, ["x", "u"], treatment_column="received"
            ),
        }
        est = fns[method]()
        assert est.point == pytest.approx(1.0, abs=0.15), (
            f"{method} should recover the truth when given u"
        )

    def test_aipw_assumptions_state_that_double_robustness_does_not_help(self):
        s = make_confounded("unobserved", 5_000, seed=0)
        est = estimate_aipw(s.data, ESTIMAND, ["x"], treatment_column="received")
        assert any("does NOT relax" in a for a in est.assumptions)


class TestTrimming:
    """Trimming must be explicit, and must announce that it changed the estimand."""

    def test_trimming_is_off_by_default(self):
        s = make_confounded("selection", 5_000, strength=1.2, seed=0)
        est = estimate_ipw(s.data, ESTIMAND, ["x"], treatment_column="received")
        assert est.diagnostics["n_trimmed"] == 0.0
        assert not any("TRIMMED" in a for a in est.assumptions)

    def test_trimming_records_the_estimand_change(self):
        s = make_confounded("selection", 20_000, strength=2.0, seed=0)
        est = estimate_ipw(s.data, ESTIMAND, ["x"], treatment_column="received", trim=0.1)
        assert est.diagnostics["n_trimmed"] > 0
        assert any("TRIMMED" in a for a in est.assumptions)
        assert any("NOT the population ATE" in a for a in est.assumptions)
        assert est.estimand.population == "overlap region"

    def test_invalid_trim_raises(self):
        s = make_confounded("selection", 2_000, seed=0)
        with pytest.raises(ValueError, match=r"trim must be in \(0, 0.5\)"):
            estimate_ipw(s.data, ESTIMAND, ["x"], treatment_column="received", trim=0.7)


class TestEstimandTargets:
    def test_ipw_and_aipw_declare_the_ate(self):
        s = make_randomised(2_000, seed=0)
        for est in (
            estimate_ipw(s.data, ESTIMAND, ["x"], treatment_column="received"),
            estimate_aipw(s.data, ESTIMAND, ["x"], treatment_column="received"),
        ):
            assert est.estimand.target is EstimandTarget.ATE


class TestEValue:
    """Hand-computable: E = RR + sqrt(RR(RR-1))."""

    def test_risk_ratio_of_two(self):
        # 2 + sqrt(2*1) = 2 + 1.41421 = 3.41421
        assert e_value(2.0).point == pytest.approx(3.4142135624, abs=1e-8)

    def test_null_result_needs_no_confounding(self):
        assert e_value(1.0).point == pytest.approx(1.0)

    def test_protective_and_harmful_effects_are_symmetric(self):
        """Explaining away RR=0.5 takes the same strength as RR=2."""
        assert e_value(0.5).point == pytest.approx(e_value(2.0).point)

    def test_larger_effects_need_stronger_confounders(self):
        values = [e_value(rr).point for rr in (1.1, 1.5, 2.0, 5.0)]
        assert values == sorted(values)

    def test_interval_crossing_the_null_gives_an_e_value_of_one(self):
        """If the interval already includes no effect, no confounding is needed."""
        assert e_value(1.5, ci_bound=0.9).bound == pytest.approx(1.0)

    def test_bound_is_smaller_than_the_point_e_value(self):
        ev = e_value(2.0, ci_bound=1.4)
        assert ev.bound is not None
        assert ev.bound < ev.point

    def test_fragility_flag(self):
        assert e_value(1.05, ci_bound=1.01).is_fragile
        assert not e_value(5.0, ci_bound=4.0).is_fragile

    def test_validation(self):
        with pytest.raises(ValueError, match="risk ratio must be positive"):
            e_value(0.0)
        with pytest.raises(ValueError, match="ci_bound must be positive"):
            e_value(2.0, ci_bound=-1.0)

    def test_describe_explains_what_the_number_means(self):
        text = e_value(2.0, ci_bound=1.4).describe()
        assert "BOTH treatment and outcome" in text


class TestEValueFromDifference:
    def test_converts_a_risk_difference(self):
        # 19% -> 18.2% is RR = 0.182/0.19 = 0.9579
        ev = e_value_from_difference(-0.008, baseline_rate=0.19)
        assert ev.risk_ratio == pytest.approx(0.182 / 0.19, abs=1e-9)

    def test_a_small_difference_gives_a_modest_e_value(self):
        """The honest reading: 0.8pp off a 19% base is a weak ratio, so a weak E-value."""
        ev = e_value_from_difference(-0.008, baseline_rate=0.19)
        assert ev.point < 1.6

    def test_impossible_difference_raises(self):
        with pytest.raises(ValueError, match="outside"):
            e_value_from_difference(0.9, baseline_rate=0.19)

    def test_bad_baseline_raises(self):
        with pytest.raises(ValueError, match="baseline_rate must be in"):
            e_value_from_difference(0.01, baseline_rate=0.0)


class TestScoring:
    def test_scores_a_perfect_estimator(self):
        from gatekeeper.benchmark.scoring import score_estimates

        estimates = [
            estimate_naive_difference(
                make_randomised(3_000, true_ate=1.0, seed=s).data,
                ESTIMAND,
                treatment_column="received",
            )
            for s in range(10)
        ]
        score = score_estimates(estimates, 1.0)
        assert abs(score.bias) < 0.1
        assert not score.is_badly_biased
        assert score.coverage > 0.7

    def test_detects_a_biased_estimator(self):
        from gatekeeper.benchmark.scoring import score_estimates

        estimates = [
            estimate_naive_difference(
                make_confounded("selection", 3_000, true_ate=1.0, strength=1.2, seed=s).data,
                ESTIMAND,
                treatment_column="received",
            )
            for s in range(10)
        ]
        score = score_estimates(estimates, 1.0)
        assert score.is_badly_biased
        assert score.coverage < 0.2
        assert not score.coverage_is_nominal

    def test_bias_in_se_units_is_scale_free(self):
        from gatekeeper.benchmark.scoring import score_estimates

        estimates = [
            estimate_naive_difference(
                make_randomised(3_000, true_ate=1.0, seed=s).data,
                ESTIMAND,
                treatment_column="received",
            )
            for s in range(8)
        ]
        score = score_estimates(estimates, 1.0)
        assert score.bias_in_se_units == pytest.approx(abs(score.bias) / score.mean_se, rel=1e-9)

    def test_too_few_replications_raises(self):
        from gatekeeper.benchmark.scoring import score_estimates

        est = estimate_naive_difference(
            make_randomised(1_000, seed=0).data, ESTIMAND, treatment_column="received"
        )
        with pytest.raises(InsufficientData, match=">= 3 replications"):
            score_estimates([est, est], 1.0)

    def test_describe_is_readable(self):
        from gatekeeper.benchmark.scoring import score_estimates

        estimates = [
            estimate_naive_difference(
                make_randomised(2_000, true_ate=1.0, seed=s).data,
                ESTIMAND,
                treatment_column="received",
            )
            for s in range(6)
        ]
        text = score_estimates(estimates, 1.0).describe()
        assert "bias" in text
        assert "coverage" in text


class TestOutcomeRegressionStandardError:
    """Regression test for a serious bug.

    The original implementation used ``(mu1 - mu0).std() / sqrt(n)``, which measures how
    much the predicted contrast varies ACROSS UNITS rather than the uncertainty of the
    estimated average. With a homogeneous effect it collapses toward zero, and the
    benchmark showed 2.5% coverage with |bias|/se above 1000.
    """

    def test_standard_error_is_not_absurdly_small(self):
        s = make_randomised(4_000, true_ate=1.0, seed=0)
        est = estimate_outcome_regression(s.data, ESTIMAND, ["x"], treatment_column="received")
        assert est.se is not None
        # A sane SE at n=4,000 with unit-variance noise is order 0.03, not 1e-4.
        assert est.se > 0.005

    def test_standard_error_is_comparable_to_other_estimators(self):
        s = make_randomised(4_000, true_ate=1.0, seed=0)
        reg = estimate_outcome_regression(s.data, ESTIMAND, ["x"], treatment_column="received")
        aipw = estimate_aipw(s.data, ESTIMAND, ["x"], treatment_column="received")
        assert reg.se is not None and aipw.se is not None
        assert reg.se == pytest.approx(aipw.se, rel=0.5)

    def test_standard_error_shrinks_as_sqrt_n(self):
        small = estimate_outcome_regression(
            make_randomised(2_000, seed=0).data, ESTIMAND, ["x"], treatment_column="received"
        )
        large = estimate_outcome_regression(
            make_randomised(32_000, seed=0).data, ESTIMAND, ["x"], treatment_column="received"
        )
        assert small.se is not None and large.se is not None
        # 16x the data should roughly quarter the standard error.
        assert large.se == pytest.approx(small.se / 4.0, rel=0.35)

    def test_interval_covers_the_truth_across_seeds(self):
        covered = 0
        for seed in range(30):
            est = estimate_outcome_regression(
                make_randomised(2_000, true_ate=1.0, seed=seed).data,
                ESTIMAND,
                ["x"],
                treatment_column="received",
            )
            covered += est.ci[0] <= 1.0 <= est.ci[1]
        assert covered >= 25, f"only {covered}/30 intervals covered the truth"


def test_numpy_is_available_for_diagnostics():
    """Guard against an accidental import removal in the modules above."""
    assert np.isfinite(1.0)
