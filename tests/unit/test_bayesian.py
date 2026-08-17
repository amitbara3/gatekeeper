"""Beta-Binomial model: conjugacy, quadrature, and the interpretive distinctions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from gatekeeper.bayesian.beta_binomial import (
    JEFFREYS_PRIOR,
    UNIFORM_PRIOR,
    BetaPrior,
    compare_beta_binomial,
    estimate_beta_binomial,
    prior_sensitivity,
    prob_b_beats_a,
)
from gatekeeper.data.synthetic import make_cookie_cats_like
from gatekeeper.types import Estimand, InsufficientData, Scale


class TestPrior:
    def test_rejects_non_positive_parameters(self):
        with pytest.raises(ValueError, match="must be positive"):
            BetaPrior(0.0, 1.0)
        with pytest.raises(ValueError, match="must be positive"):
            BetaPrior(1.0, -1.0)

    def test_mean_and_prior_sample_size(self):
        p = BetaPrior(2.0, 8.0)
        assert p.mean == pytest.approx(0.2)
        assert p.prior_sample_size == pytest.approx(10.0)

    def test_uniform_and_jeffreys(self):
        assert UNIFORM_PRIOR.mean == pytest.approx(0.5)
        assert UNIFORM_PRIOR.prior_sample_size == pytest.approx(2.0)
        assert JEFFREYS_PRIOR.prior_sample_size == pytest.approx(1.0)


class TestConjugacy:
    def test_update_is_the_textbook_formula(self):
        """Beta(a,b) + s successes in n trials -> Beta(a+s, b+n-s)."""
        post = BetaPrior(2.0, 3.0).update(successes=7, trials=10)
        assert post.alpha == pytest.approx(9.0)
        assert post.beta == pytest.approx(6.0)

    def test_posterior_mean_lies_between_prior_mean_and_sample_rate(self):
        prior = BetaPrior(2.0, 8.0)  # mean 0.2
        post = prior.update(successes=90, trials=100)  # sample rate 0.9
        assert prior.mean < post.mean < 0.9

    def test_posterior_concentrates_as_data_grows(self):
        small = JEFFREYS_PRIOR.update(19, 100)
        large = JEFFREYS_PRIOR.update(1_900, 10_000)
        assert large.sd < small.sd
        assert large.mean == pytest.approx(0.19, abs=0.01)

    def test_posterior_mode_matches_the_analytic_formula(self):
        post = BetaPrior(2.0, 2.0).update(6, 10)  # Beta(8, 6)
        assert post.mode == pytest.approx((8 - 1) / (8 + 6 - 2))

    def test_mode_falls_back_to_mean_for_flat_posteriors(self):
        post = BetaPrior(0.5, 0.5).update(0, 0)  # Beta(0.5, 0.5): no interior mode
        assert post.mode == pytest.approx(post.mean)

    def test_invalid_counts_raise(self):
        with pytest.raises(ValueError, match="0 <= successes <= trials"):
            JEFFREYS_PRIOR.update(successes=11, trials=10)
        with pytest.raises(ValueError, match="0 <= successes <= trials"):
            JEFFREYS_PRIOR.update(successes=-1, trials=10)

    def test_credible_interval_brackets_the_mean(self):
        post = JEFFREYS_PRIOR.update(190, 1_000)
        lo, hi = post.credible_interval(0.95)
        assert lo < post.mean < hi

    def test_credible_interval_matches_scipy_beta_quantiles(self):
        post = JEFFREYS_PRIOR.update(190, 1_000)
        lo, hi = post.credible_interval(0.90)
        assert lo == pytest.approx(stats.beta.ppf(0.05, post.alpha, post.beta))
        assert hi == pytest.approx(stats.beta.ppf(0.95, post.alpha, post.beta))

    def test_higher_level_gives_wider_interval(self):
        post = JEFFREYS_PRIOR.update(190, 1_000)
        narrow = post.credible_interval(0.80)
        wide = post.credible_interval(0.99)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_bad_level_raises(self):
        with pytest.raises(ValueError, match="level must be in"):
            JEFFREYS_PRIOR.update(5, 10).credible_interval(1.5)


class TestProbBBeatsA:
    """The headline number, computed by quadrature rather than sampled."""

    def test_identical_posteriors_give_one_half(self):
        a = JEFFREYS_PRIOR.update(200, 1_000)
        b = JEFFREYS_PRIOR.update(200, 1_000)
        assert prob_b_beats_a(a, b) == pytest.approx(0.5, abs=1e-6)

    def test_matches_monte_carlo(self):
        """Cross-check quadrature against 10 million draws."""
        rng = np.random.default_rng(0)
        for s_a, n_a, s_b, n_b in [
            (200, 1_000, 250, 1_000),
            (20, 100, 30, 100),
            (8_550, 45_000, 8_190, 45_000),
            (5, 50, 40, 50),
        ]:
            a = JEFFREYS_PRIOR.update(s_a, n_a)
            b = JEFFREYS_PRIOR.update(s_b, n_b)
            exact = prob_b_beats_a(a, b)
            draws = 10_000_000
            mc = float(np.mean(rng.beta(b.alpha, b.beta, draws) > rng.beta(a.alpha, a.beta, draws)))
            assert exact == pytest.approx(mc, abs=0.001), (
                f"quadrature {exact:.6f} vs Monte Carlo {mc:.6f} at ({s_a}/{n_a} vs {s_b}/{n_b})"
            )

    def test_handles_extremely_peaked_posteriors(self):
        """The regression this integration range exists for.

        At n=450,000 the posterior sd is ~0.0006, so the density occupies well under 1%
        of [0, 1]. Integrating naively over the unit interval lets adaptive quadrature
        step over the mass entirely.
        """
        a = JEFFREYS_PRIOR.update(85_500, 450_000)
        b = JEFFREYS_PRIOR.update(81_900, 450_000)
        p = prob_b_beats_a(a, b)
        assert 0.0 <= p <= 1.0
        # Treatment is clearly worse at this sample size.
        assert p < 1e-6

    def test_is_deterministic(self):
        a = JEFFREYS_PRIOR.update(200, 1_000)
        b = JEFFREYS_PRIOR.update(250, 1_000)
        assert prob_b_beats_a(a, b) == prob_b_beats_a(a, b)

    def test_complementary_under_swap(self):
        a = JEFFREYS_PRIOR.update(200, 1_000)
        b = JEFFREYS_PRIOR.update(250, 1_000)
        assert prob_b_beats_a(a, b) + prob_b_beats_a(b, a) == pytest.approx(1.0, abs=1e-6)

    def test_clear_winner_approaches_one(self):
        a = JEFFREYS_PRIOR.update(100, 10_000)
        b = JEFFREYS_PRIOR.update(2_000, 10_000)
        assert prob_b_beats_a(a, b) > 0.9999


class TestComparison:
    def test_recovers_a_known_lift(self):
        c = compare_beta_binomial(2_000, 10_000, 2_500, 10_000, seed=1)
        assert c.lift_absolute == pytest.approx(0.05, abs=0.002)
        assert c.prob_treatment_better > 0.999
        lo, hi = c.lift_credible_interval
        assert lo <= c.lift_absolute <= hi
        assert lo > 0

    def test_no_difference_gives_probability_near_one_half(self):
        c = compare_beta_binomial(2_000, 10_000, 2_000, 10_000, seed=2)
        assert c.prob_treatment_better == pytest.approx(0.5, abs=0.01)
        lo, hi = c.lift_credible_interval
        assert lo < 0 < hi

    def test_probabilities_are_complementary(self):
        c = compare_beta_binomial(2_000, 10_000, 2_300, 10_000, seed=3)
        assert c.prob_treatment_better + c.prob_control_better == pytest.approx(1.0)

    def test_expected_losses_are_non_negative_and_asymmetric(self):
        c = compare_beta_binomial(2_000, 10_000, 2_500, 10_000, seed=4)
        assert c.expected_loss_ship >= 0
        assert c.expected_loss_keep >= 0
        # Treatment is clearly better, so keeping control is the costly mistake.
        assert c.expected_loss_keep > c.expected_loss_ship

    def test_credible_interval_narrows_with_more_data(self):
        small = compare_beta_binomial(200, 1_000, 250, 1_000, seed=5)
        large = compare_beta_binomial(20_000, 100_000, 25_000, 100_000, seed=5)
        small_width = small.lift_credible_interval[1] - small.lift_credible_interval[0]
        large_width = large.lift_credible_interval[1] - large.lift_credible_interval[0]
        assert large_width < small_width

    def test_reproducible_under_the_same_seed(self):
        a = compare_beta_binomial(200, 1_000, 250, 1_000, seed=7)
        b = compare_beta_binomial(200, 1_000, 250, 1_000, seed=7)
        assert a.lift_credible_interval == b.lift_credible_interval
        assert a.expected_loss_ship == b.expected_loss_ship

    def test_validation(self):
        with pytest.raises(InsufficientData, match="at least one trial"):
            compare_beta_binomial(0, 0, 5, 10)
        with pytest.raises(ValueError, match="n_samples must be >= 1000"):
            compare_beta_binomial(5, 10, 6, 10, n_samples=10)
        with pytest.raises(ValueError, match="credible_level"):
            compare_beta_binomial(5, 10, 6, 10, credible_level=0.0)

    def test_describe_is_readable(self):
        text = compare_beta_binomial(2_000, 10_000, 2_500, 10_000, seed=8).describe()
        assert "P(treatment > control)" in text
        assert "credible interval" in text
        assert "expected loss" in text


class TestExpectedLossDecision:
    def test_ships_a_clear_winner(self):
        c = compare_beta_binomial(2_000, 10_000, 2_500, 10_000, seed=9)
        assert c.decision_at(loss_threshold=0.001) == "ship"

    def test_keeps_control_when_treatment_is_clearly_worse(self):
        c = compare_beta_binomial(2_500, 10_000, 2_000, 10_000, seed=10)
        assert c.decision_at(loss_threshold=0.001) == "keep"

    def test_undecided_when_both_losses_exceed_the_threshold(self):
        """A genuinely uncertain result should ask for more data, not guess."""
        c = compare_beta_binomial(50, 200, 60, 200, seed=11)
        assert c.decision_at(loss_threshold=1e-6) == "undecided"

    def test_prefers_the_status_quo_when_the_arms_are_equivalent(self):
        """Both errors cheap => no reason to churn."""
        c = compare_beta_binomial(20_000, 100_000, 20_010, 100_000, seed=12)
        assert c.decision_at(loss_threshold=0.01) == "keep"

    def test_high_probability_of_a_tiny_effect_is_not_a_ship(self):
        """The reason expected loss beats P(B > A) as a decision rule.

        A huge sample makes P(B > A) overwhelming for an effect far too small to matter.
        Probability alone would ship it; expected loss will not, if the threshold
        reflects what is worth shipping for.
        """
        # 20.00% vs 20.13% over a million per arm: z ~ 2.3, so the probability is
        # overwhelming while the effect itself is 0.13pp.
        c = compare_beta_binomial(200_000, 1_000_000, 201_300, 1_000_000, seed=13)
        assert c.prob_treatment_better > 0.98
        # The whole effect is ~0.0013, so at a 0.01 threshold nothing is at stake and
        # expected loss correctly declines to ship.
        assert c.lift_absolute < 0.01
        assert c.decision_at(loss_threshold=0.01) == "keep"

    def test_rejects_non_positive_threshold(self):
        c = compare_beta_binomial(200, 1_000, 250, 1_000, seed=14)
        with pytest.raises(ValueError, match="loss_threshold must be positive"):
            c.decision_at(loss_threshold=0.0)


class TestPriorSensitivity:
    def test_prior_barely_matters_at_large_n(self):
        results = prior_sensitivity(8_550, 45_000, 8_190, 45_000, seed=15)
        probs = [r.prob_treatment_better for r in results.values()]
        # 45,000 observations per arm swamp every prior tried.
        assert max(probs) - min(probs) < 0.02, f"probabilities across priors: {probs}"

    def test_prior_dominates_at_small_n(self):
        """The other half of the claim, which is why it has to be checked not assumed."""
        results = prior_sensitivity(3, 10, 6, 10, seed=16)
        probs = [r.prob_treatment_better for r in results.values()]
        assert max(probs) - min(probs) > 0.05, f"probabilities across priors: {probs}"

    def test_default_priors_are_covered(self):
        results = prior_sensitivity(200, 1_000, 250, 1_000, seed=17)
        assert set(results) == {"jeffreys", "uniform", "sceptical_50_50", "strong_low_rate"}

    def test_custom_priors(self):
        results = prior_sensitivity(200, 1_000, 250, 1_000, priors=(UNIFORM_PRIOR,), seed=18)
        assert set(results) == {"uniform"}


class TestEstimateWrapper:
    def test_returns_a_populated_estimate(self):
        exp = make_cookie_cats_like(n=20_000, seed=19)
        est = estimate_beta_binomial(exp.data, Estimand(outcome="retention_7", treatment="version"))
        assert est.method == "beta_binomial_posterior"
        assert est.assumptions
        assert est.ci[0] <= est.point <= est.ci[1]
        assert "prob_treatment_better" in est.diagnostics

    def test_p_value_is_deliberately_absent(self):
        """There is no p-value in this framework; filling the field would mislead."""
        exp = make_cookie_cats_like(n=10_000, seed=20)
        est = estimate_beta_binomial(exp.data, Estimand(outcome="retention_7", treatment="version"))
        assert est.p_value is None
        assert est.se is None

    def test_assumptions_state_the_two_interpretive_traps(self):
        exp = make_cookie_cats_like(n=5_000, seed=21)
        est = estimate_beta_binomial(exp.data, Estimand(outcome="retention_1", treatment="version"))
        text = " ".join(est.assumptions)
        assert "NOT a confidence interval" in text
        assert "NOT 1 minus" in text

    def test_assumptions_quantify_the_prior_against_the_data(self):
        exp = make_cookie_cats_like(n=5_000, seed=22)
        est = estimate_beta_binomial(exp.data, Estimand(outcome="retention_1", treatment="version"))
        assert any("worth 1 observations against 5,000" in a for a in est.assumptions)

    def test_recovers_the_known_true_effect(self):
        exp = make_cookie_cats_like(n=200_000, seed=23, retention_7_effect=-0.02)
        est = estimate_beta_binomial(exp.data, Estimand(outcome="retention_7", treatment="version"))
        assert est.ci[0] <= exp.true_effect("retention_7") <= est.ci[1]

    def test_non_binary_metric_is_rejected(self):
        exp = make_cookie_cats_like(n=2_000, seed=24)
        with pytest.raises(ValueError, match="not binary"):
            estimate_beta_binomial(
                exp.data, Estimand(outcome="sum_gamerounds", treatment="version")
            )

    def test_relative_scale_is_refused(self):
        exp = make_cookie_cats_like(n=2_000, seed=25)
        with pytest.raises(NotImplementedError, match="relative-scale"):
            estimate_beta_binomial(
                exp.data,
                Estimand(outcome="retention_7", treatment="version", scale=Scale.RELATIVE),
            )


class TestAgreementWithFrequentist:
    """At large n with a weak prior the two frameworks should land in the same place.

    They answer different questions, but the numbers must not diverge -- if they did, one
    of the implementations would be wrong.
    """

    def test_point_estimates_agree(self):
        from gatekeeper.frequentist.proportions import two_proportion_test

        s_c, n_c, s_t, n_t = 8_550, 45_000, 8_190, 45_000
        bayes = compare_beta_binomial(s_c, n_c, s_t, n_t, seed=26)
        freq = two_proportion_test(s_c, n_c, s_t, n_t)
        assert bayes.lift_absolute == pytest.approx(freq.point, abs=1e-4)

    def test_interval_widths_agree(self):
        from gatekeeper.frequentist.proportions import two_proportion_test

        s_c, n_c, s_t, n_t = 8_550, 45_000, 8_190, 45_000
        bayes = compare_beta_binomial(s_c, n_c, s_t, n_t, seed=27)
        freq = two_proportion_test(s_c, n_c, s_t, n_t)
        bayes_width = bayes.lift_credible_interval[1] - bayes.lift_credible_interval[0]
        freq_width = freq.ci[1] - freq.ci[0]
        assert bayes_width == pytest.approx(freq_width, rel=0.05)

    def test_probability_is_not_one_minus_the_p_value(self):
        """They are close here by arithmetic coincidence, not correspondence.

        Documented as a test so nobody later "simplifies" one into the other.
        """
        from gatekeeper.frequentist.proportions import two_proportion_test

        s_c, n_c, s_t, n_t = 500, 5_000, 560, 5_000
        bayes = compare_beta_binomial(s_c, n_c, s_t, n_t, seed=28)
        freq = two_proportion_test(s_c, n_c, s_t, n_t)
        # One-sided p is the closest frequentist analogue, and even that only matches
        # approximately -- and means something entirely different.
        one_sided_p = freq.p_value / 2
        assert bayes.prob_treatment_better == pytest.approx(1 - one_sided_p, abs=0.02)
        # But it is emphatically NOT 1 - two-sided p.
        assert bayes.prob_treatment_better != pytest.approx(1 - freq.p_value, abs=1e-6)
