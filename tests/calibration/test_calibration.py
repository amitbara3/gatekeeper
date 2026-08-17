"""Calibration: p-value uniformity under the null, and confidence-interval coverage.

**This is the layer that matters.** A fixture proves one case. A property test proves an
invariant. A reference cross-check proves we agree with somebody else -- who may be
solving a subtly different problem. None of them establish the thing an estimator
actually claims:

- under a true null, a p-value is uniform on [0, 1], so ``P(p < alpha) = alpha``;
- a nominal 95% interval contains the true parameter 95% of the time.

Both are checked by simulating from a known data-generating process, which is the only
way to test a statistical claim rather than an implementation detail.

Marked ``slow`` and run nightly rather than per-push (Architecture §8). Every test is
seeded, so a failure is reproducible (R4.2).

R4.7: a failure here is a bug until proven otherwise. Do not widen a tolerance to get
green.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy import stats

from gatekeeper.frequentist.bootstrap import bootstrap_mean_difference
from gatekeeper.frequentist.means import welch_test
from gatekeeper.frequentist.proportions import two_proportion_test

pytestmark = pytest.mark.slow

N_SIMS = 2_000
"""Enough to resolve a 95% coverage rate to about +/-0.5pp (1 SE)."""

COVERAGE_LO, COVERAGE_HI = 0.93, 0.97
"""Acceptance band for nominal 95% coverage. At 2,000 sims, 1 SE is ~0.005, so this
band is roughly +/-4 SE -- wide enough not to flake, tight enough to catch a real
mis-calibration (a variance error of 10% moves coverage by several points)."""


def _uniformity_p(p_values: np.ndarray) -> float:
    """One-sample KS test of ``p_values`` against Uniform(0, 1).

    **Only valid for a continuous test statistic.** See :func:`assert_level_is_correct`
    for the discrete case and why KS cannot be used there.
    """
    return float(stats.kstest(p_values, "uniform").pvalue)


LEVELS = (0.01, 0.05, 0.10)
"""Significance levels at which the test's actual error rate is checked."""


def assert_level_is_correct(p_values: np.ndarray, tolerance_sigma: float = 4.0) -> None:
    """Assert ``P(p <= a) ~= a`` at each level in :data:`LEVELS`.

    This is the operational definition of a calibrated p-value, and unlike a KS test it
    is valid whether the statistic is continuous or discrete.

    **Why this and not KS, for a discrete statistic.** The two-proportion z-test is
    computed from binomial counts, so its p-value distribution has atoms: at a 0.5 base
    rate with n=800, 2,000 simulations produce only ~973 distinct p-values, and a single
    value can carry over 2% of the mass. ``scipy.stats.kstest`` compares against a
    *continuous* uniform CDF and will reject such a sample systematically -- not because
    the test is mis-calibrated but because it is being measured with the wrong
    instrument. Substituting this check is a correction of method, **not** a loosened
    tolerance (R4.7).

    Parameters
    ----------
    p_values
        Simulated p-values, all drawn under a true null.
    tolerance_sigma
        How many binomial standard errors of slack to allow. The SE at level ``a`` over
        ``n`` simulations is ``sqrt(a(1-a)/n)``, so the band tightens automatically as
        the simulation count grows -- there is no hand-tuned constant to drift.
    """
    n = p_values.size
    for level in LEVELS:
        observed = float(np.mean(p_values <= level))
        se = float(np.sqrt(level * (1.0 - level) / n))
        assert abs(observed - level) <= tolerance_sigma * se, (
            f"error rate at alpha={level} was {observed:.4f}, off nominal by "
            f"{abs(observed - level) / se:.1f} SE (limit {tolerance_sigma}); "
            f"n={n} simulations"
        )


def assert_uniform_across_seeds(
    p_value_fn: Callable[[np.random.Generator], np.ndarray],
    n_seeds: int = 8,
    base_seed: int = 1_000,
) -> None:
    """Assert KS uniformity is not *systematically* violated across seeds.

    For a **continuous** statistic, KS is the right test -- but a single run at
    alpha=0.05 fails 5% of the time by construction, so a suite containing several such
    tests is guaranteed to flake. Welch is a live example: over 12 seeds its KS p-values
    ranged from 0.014 to 0.87 with a median of 0.32, and 3 of 36 runs fell below 0.05 --
    exactly the ~5% expected of a correctly calibrated test.

    Under calibration each seed's KS p-value is itself Uniform(0, 1), so the *median*
    over ``n_seeds`` seeds exceeds 0.05 with overwhelming probability (it would take half
    the seeds failing). Real mis-calibration pushes every KS p-value toward zero and
    drags the median down. Testing the median therefore keeps full sensitivity to a
    systematic problem while being effectively immune to one unlucky seed.

    Parameters
    ----------
    p_value_fn
        Callable taking a ``numpy.random.Generator`` and returning an array of
        simulated p-values.
    """
    ks_p = np.array(
        [_uniformity_p(p_value_fn(np.random.default_rng(base_seed + s))) for s in range(n_seeds)]
    )
    median = float(np.median(ks_p))
    assert median > 0.05, (
        f"KS uniformity is systematically violated: median KS p over {n_seeds} seeds "
        f"was {median:.4f} (values: {np.round(ks_p, 4).tolist()}). A single low value is "
        "seed noise; a low median is mis-calibration."
    )


# ---------------------------------------------------------------------------
# Two-proportion z-test
# ---------------------------------------------------------------------------


def _null_proportion_p_values(base_rate: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate ``N_SIMS`` two-proportion p-values under a true null."""
    out = np.empty(N_SIMS)
    for i in range(N_SIMS):
        s_c = int(rng.binomial(n, base_rate))
        s_t = int(rng.binomial(n, base_rate))
        out[i] = two_proportion_test(s_c, n, s_t, n, warn_small=False).p_value
    return out


class TestTwoProportionCalibration:
    @pytest.mark.parametrize(("base_rate", "n"), [(0.19, 2_000), (0.50, 800), (0.05, 5_000)])
    def test_error_rate_matches_alpha_at_every_level(self, base_rate: float, n: int):
        """The correct calibration check for a discrete statistic.

        Not a KS test: see :func:`assert_level_is_correct` for why KS is invalid here.
        """
        p_values = _null_proportion_p_values(base_rate, n, np.random.default_rng(20260817))
        assert_level_is_correct(p_values)

    @pytest.mark.parametrize(("base_rate", "n"), [(0.50, 800), (0.19, 2_000)])
    def test_the_p_value_distribution_really_is_discrete(self, base_rate: float, n: int):
        """Verify the premise behind skipping KS, rather than asserting it in prose.

        If this ever fails -- if the p-values become effectively continuous -- then KS
        would be the appropriate instrument and this module should switch back to it.
        """
        p_values = _null_proportion_p_values(base_rate, n, np.random.default_rng(20260817))
        n_unique = len(np.unique(p_values))
        _, counts = np.unique(p_values, return_counts=True)
        assert n_unique < N_SIMS, (
            f"expected atoms from binomial counts but got {n_unique} distinct values "
            f"in {N_SIMS} draws"
        )
        largest_atom = counts.max() / N_SIMS
        assert largest_atom > 0.005, (
            f"largest atom holds only {largest_atom:.4%} of the mass; the distribution "
            "may be near-continuous, in which case prefer a KS test"
        )

    @pytest.mark.parametrize(("base_rate", "n"), [(0.19, 2_000), (0.50, 800)])
    def test_false_positive_rate_matches_alpha(self, base_rate: float, n: int):
        rng = np.random.default_rng(11)
        rejections = 0
        for _ in range(N_SIMS):
            s_c = int(rng.binomial(n, base_rate))
            s_t = int(rng.binomial(n, base_rate))
            rejections += two_proportion_test(s_c, n, s_t, n, warn_small=False).p_value < 0.05
        rate = rejections / N_SIMS
        # Binomial SE at 0.05 over 2000 sims is ~0.0049; allow ~4 SE.
        assert rate == pytest.approx(0.05, abs=0.02), f"false positive rate {rate:.4f}"

    @pytest.mark.parametrize(
        ("base_rate", "effect", "n"),
        [(0.19, -0.02, 3_000), (0.50, 0.05, 1_000), (0.10, 0.03, 2_000)],
    )
    def test_ci_covers_the_true_effect_at_the_nominal_rate(
        self, base_rate: float, effect: float, n: int
    ):
        rng = np.random.default_rng(22)
        covered = 0
        for _ in range(N_SIMS):
            s_c = int(rng.binomial(n, base_rate))
            s_t = int(rng.binomial(n, base_rate + effect))
            r = two_proportion_test(s_c, n, s_t, n, warn_small=False)
            covered += r.ci[0] <= effect <= r.ci[1]
        coverage = covered / N_SIMS
        assert COVERAGE_LO <= coverage <= COVERAGE_HI, (
            f"95% CI covered the true effect {coverage:.1%} of the time "
            f"(want {COVERAGE_LO:.0%}-{COVERAGE_HI:.0%}) at base_rate={base_rate}, "
            f"effect={effect}, n={n}"
        )

    def test_relative_scale_ci_coverage(self):
        from gatekeeper.types import Scale

        base_rate, effect, n = 0.20, 0.04, 3_000
        true_relative = (base_rate + effect) / base_rate - 1.0
        rng = np.random.default_rng(33)
        covered = 0
        for _ in range(N_SIMS):
            s_c = int(rng.binomial(n, base_rate))
            s_t = int(rng.binomial(n, base_rate + effect))
            r = two_proportion_test(s_c, n, s_t, n, scale=Scale.RELATIVE, warn_small=False)
            covered += r.ci[0] <= true_relative <= r.ci[1]
        coverage = covered / N_SIMS
        assert COVERAGE_LO <= coverage <= COVERAGE_HI, (
            f"relative-scale coverage {coverage:.1%}; the log-ratio interval is off"
        )

    @pytest.mark.parametrize("alpha", [0.01, 0.10])
    def test_coverage_tracks_the_requested_level(self, alpha: float):
        base_rate, effect, n = 0.30, 0.03, 2_000
        rng = np.random.default_rng(44)
        covered = 0
        for _ in range(N_SIMS):
            s_c = int(rng.binomial(n, base_rate))
            s_t = int(rng.binomial(n, base_rate + effect))
            r = two_proportion_test(s_c, n, s_t, n, alpha=alpha, warn_small=False)
            covered += r.ci[0] <= effect <= r.ci[1]
        coverage = covered / N_SIMS
        assert coverage == pytest.approx(1 - alpha, abs=0.02), (
            f"nominal {1 - alpha:.0%} interval achieved {coverage:.1%}"
        )


# ---------------------------------------------------------------------------
# Welch's t-test
# ---------------------------------------------------------------------------


class TestWelchCalibration:
    @pytest.mark.parametrize(
        ("sd_c", "sd_t", "n_c", "n_t"),
        [
            (1.0, 1.0, 100, 100),
            (1.0, 4.0, 100, 100),  # unequal variance -- Welch's whole reason to exist
            (1.0, 4.0, 40, 200),  # unequal variance AND unequal n
        ],
    )
    def test_p_values_are_uniform_under_the_null(
        self, sd_c: float, sd_t: float, n_c: int, n_t: int
    ):
        # Welch's p-value IS continuous, so KS is the right instrument here -- but a
        # single-seed KS test at alpha=0.05 fails 5% of the time by construction.
        # Checking the median across seeds stays sensitive to systematic
        # mis-calibration while being immune to one unlucky draw.
        assert_uniform_across_seeds(
            lambda rng: np.array(
                [
                    welch_test(rng.normal(0.0, sd_c, n_c), rng.normal(0.0, sd_t, n_t)).p_value
                    for _ in range(N_SIMS)
                ]
            )
        )

    @pytest.mark.parametrize(
        ("sd_c", "sd_t", "n_c", "n_t"),
        [(1.0, 1.0, 100, 100), (1.0, 4.0, 100, 100), (1.0, 4.0, 40, 200)],
    )
    def test_error_rate_matches_alpha_at_every_level(
        self, sd_c: float, sd_t: float, n_c: int, n_t: int
    ):
        rng = np.random.default_rng(55)
        p_values = np.array(
            [
                welch_test(rng.normal(0.0, sd_c, n_c), rng.normal(0.0, sd_t, n_t)).p_value
                for _ in range(N_SIMS)
            ]
        )
        assert_level_is_correct(p_values)

    def test_unequal_variance_is_where_a_pooled_test_would_fail(self):
        """Demonstrates the cost of R1.12's forbidden default.

        With unequal variances *and* unequal group sizes, the equal-variance t-test is
        mis-calibrated while Welch is not. This test asserts Welch's correctness and
        documents the pooled test's failure -- the reason pooled is not our default.
        """
        rng = np.random.default_rng(66)
        n_c, n_t, sd_c, sd_t = 20, 200, 1.0, 6.0
        welch_p, pooled_p = np.empty(N_SIMS), np.empty(N_SIMS)
        for i in range(N_SIMS):
            c = rng.normal(0.0, sd_c, n_c)
            t = rng.normal(0.0, sd_t, n_t)
            welch_p[i] = welch_test(c, t).p_value
            pooled_p[i] = float(stats.ttest_ind(t, c, equal_var=True).pvalue)

        assert _uniformity_p(welch_p) > 0.05, "Welch should be calibrated here"
        welch_fpr = float(np.mean(welch_p < 0.05))
        pooled_fpr = float(np.mean(pooled_p < 0.05))
        assert welch_fpr == pytest.approx(0.05, abs=0.02)
        # The pooled test's error rate is visibly wrong in this regime.
        assert abs(pooled_fpr - 0.05) > abs(welch_fpr - 0.05)

    @pytest.mark.parametrize(
        ("effect", "sd_c", "sd_t", "n_c", "n_t"),
        [(0.5, 1.0, 1.0, 100, 100), (2.0, 1.0, 5.0, 80, 120)],
    )
    def test_ci_coverage(self, effect: float, sd_c: float, sd_t: float, n_c: int, n_t: int):
        rng = np.random.default_rng(77)
        covered = 0
        for _ in range(N_SIMS):
            c = rng.normal(0.0, sd_c, n_c)
            t = rng.normal(effect, sd_t, n_t)
            r = welch_test(c, t)
            covered += r.ci[0] <= effect <= r.ci[1]
        coverage = covered / N_SIMS
        assert COVERAGE_LO <= coverage <= COVERAGE_HI, f"Welch coverage {coverage:.1%}"

    def test_coverage_holds_on_skewed_data_at_large_n(self):
        """The CLT rescues Welch on a skewed metric -- but only at adequate n.

        Documents when the t-test is defensible for something like sum_gamerounds.
        """
        rng = np.random.default_rng(88)
        n = 2_000
        covered = 0
        sims = 1_000
        for _ in range(sims):
            c = rng.lognormal(1.0, 1.4, n)
            t = rng.lognormal(1.0, 1.4, n)
            r = welch_test(c, t)
            covered += r.ci[0] <= 0.0 <= r.ci[1]
        coverage = covered / sims
        assert COVERAGE_LO <= coverage <= COVERAGE_HI, (
            f"coverage on lognormal data at n={n} was {coverage:.1%}"
        )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


class TestBootstrapCalibration:
    """Fewer sims: each one runs a full bootstrap, so cost is sims x resamples."""

    SIMS = 600
    RESAMPLES = 1_200
    BAND_LO, BAND_HI = 0.92, 0.98
    """Wider than the analytic band: with 600 sims, 1 SE is ~0.9pp, so +/-3 SE needs
    about +/-2.7pp. This is a Monte Carlo precision limit, not a relaxed standard --
    the estimator is still being held to nominal 95%."""

    @pytest.mark.parametrize("method", ["percentile", "bca"])
    def test_coverage_on_normal_data(self, method: str):
        rng = np.random.default_rng(99)
        effect, n = 0.5, 300
        covered = 0
        for _ in range(self.SIMS):
            c = rng.normal(0.0, 1.0, n)
            t = rng.normal(effect, 1.0, n)
            r = bootstrap_mean_difference(c, t, n_resamples=self.RESAMPLES, method=method, rng=rng)
            covered += r.ci[0] <= effect <= r.ci[1]
        coverage = covered / self.SIMS
        assert self.BAND_LO <= coverage <= self.BAND_HI, (
            f"{method} coverage on normal data was {coverage:.1%}"
        )

    @pytest.mark.parametrize("method", ["percentile", "bca"])
    def test_coverage_on_skewed_data(self, method: str):
        """The case the bootstrap exists for -- and where BCa should earn its keep."""
        rng = np.random.default_rng(101)
        n, sigma = 400, 1.3
        # True mean of lognormal(mu, sigma) is exp(mu + sigma^2/2); identical arms
        # give a true difference of exactly zero.
        covered = 0
        for _ in range(self.SIMS):
            c = rng.lognormal(0.0, sigma, n)
            t = rng.lognormal(0.0, sigma, n)
            r = bootstrap_mean_difference(c, t, n_resamples=self.RESAMPLES, method=method, rng=rng)
            covered += r.ci[0] <= 0.0 <= r.ci[1]
        coverage = covered / self.SIMS
        assert self.BAND_LO <= coverage <= self.BAND_HI, (
            f"{method} coverage on lognormal data was {coverage:.1%}"
        )

    def test_coverage_with_a_nonzero_true_effect_on_skewed_data(self):
        rng = np.random.default_rng(202)
        n, sigma = 500, 1.0
        mu_c, mu_t = 0.0, 0.3
        true_effect = np.exp(mu_t + sigma**2 / 2) - np.exp(mu_c + sigma**2 / 2)
        covered = 0
        for _ in range(self.SIMS):
            c = rng.lognormal(mu_c, sigma, n)
            t = rng.lognormal(mu_t, sigma, n)
            r = bootstrap_mean_difference(c, t, n_resamples=self.RESAMPLES, method="bca", rng=rng)
            covered += r.ci[0] <= true_effect <= r.ci[1]
        coverage = covered / self.SIMS
        assert self.BAND_LO <= coverage <= self.BAND_HI, (
            f"BCa coverage with a true effect on skewed data was {coverage:.1%}"
        )

    def test_bootstrap_p_value_false_positive_rate(self):
        rng = np.random.default_rng(303)
        n = 300
        rejections = 0
        for _ in range(self.SIMS):
            c = rng.normal(0.0, 1.0, n)
            t = rng.normal(0.0, 1.0, n)
            r = bootstrap_mean_difference(
                c, t, n_resamples=self.RESAMPLES, method="percentile", rng=rng
            )
            rejections += r.p_value < 0.05
        rate = rejections / self.SIMS
        assert rate == pytest.approx(0.05, abs=0.03), (
            f"bootstrap achieved-significance-level FPR was {rate:.3f}"
        )


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


class TestMultiplicityCalibration:
    def test_bonferroni_controls_the_family_wise_error_rate(self):
        """With all nulls true, P(any rejection) must be <= alpha."""
        from gatekeeper.frequentist.multiplicity import correct

        rng = np.random.default_rng(404)
        m, sims = 8, 4_000
        any_rejection = 0
        for _ in range(sims):
            p = rng.uniform(0.0, 1.0, m)  # valid p-values under the null
            any_rejection += any(correct(p, method="bonferroni").rejected)
        fwer = any_rejection / sims
        assert fwer <= 0.05 + 0.01, f"Bonferroni FWER was {fwer:.4f}, above alpha"

    def test_uncorrected_testing_inflates_the_error_rate(self):
        """The problem multiplicity correction solves, measured.

        Eight independent metrics at alpha=0.05 give 1 - 0.95^8 = 33.7% chance of at
        least one false positive.
        """
        from gatekeeper.frequentist.multiplicity import correct

        rng = np.random.default_rng(505)
        m, sims = 8, 4_000
        any_rejection = 0
        for _ in range(sims):
            p = rng.uniform(0.0, 1.0, m)
            any_rejection += any(correct(p, method="none").rejected)
        rate = any_rejection / sims
        expected = 1 - 0.95**m
        assert rate == pytest.approx(expected, abs=0.03), (
            f"uncorrected family-wise error rate {rate:.3f}, expected ~{expected:.3f}"
        )

    def test_bh_controls_the_false_discovery_rate(self):
        """With a mix of true and false nulls, FDR <= alpha in expectation."""
        from gatekeeper.frequentist.multiplicity import correct

        rng = np.random.default_rng(606)
        sims = 2_000
        n_true_null, n_false_null = 8, 4
        false_discovery_proportions = []
        for _ in range(sims):
            # True nulls: uniform p-values. False nulls: strongly significant.
            p = np.concatenate([rng.uniform(0, 1, n_true_null), rng.uniform(0, 1e-4, n_false_null)])
            rejected = np.array(correct(p, alpha=0.05).rejected)
            n_rejected = rejected.sum()
            if n_rejected == 0:
                false_discovery_proportions.append(0.0)
            else:
                false_positives = rejected[:n_true_null].sum()
                false_discovery_proportions.append(false_positives / n_rejected)
        fdr = float(np.mean(false_discovery_proportions))
        assert fdr <= 0.05 + 0.01, f"BH false discovery rate was {fdr:.4f}"
