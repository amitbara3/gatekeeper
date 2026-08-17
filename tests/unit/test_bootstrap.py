"""Bootstrap intervals: percentile, BCa, and the influence-value algebra."""

from __future__ import annotations

import numpy as np
import pytest

from gatekeeper.data.synthetic import make_cookie_cats_like
from gatekeeper.frequentist.bootstrap import (
    _brute_force_influence,
    _influence_values,
    bootstrap_mean_difference,
    bootstrap_statistic,
    estimate_bootstrap,
)
from gatekeeper.types import Estimand, InsufficientData, Scale


class TestInfluenceValues:
    """The closed form is what makes BCa O(n) instead of O(n^2). Verify the algebra."""

    def test_closed_form_matches_brute_force_jackknife(self):
        rng = np.random.default_rng(1)
        c = rng.normal(5, 2, 17)
        t = rng.normal(6, 3, 23)
        np.testing.assert_allclose(
            _influence_values(c, t), _brute_force_influence(c, t), rtol=1e-10, atol=1e-10
        )

    def test_matches_brute_force_on_skewed_data(self):
        rng = np.random.default_rng(2)
        c = rng.lognormal(1.0, 1.5, 25)
        t = rng.lognormal(1.2, 1.5, 19)
        np.testing.assert_allclose(
            _influence_values(c, t), _brute_force_influence(c, t), rtol=1e-10, atol=1e-10
        )

    def test_influence_values_sum_to_zero(self):
        """Centred values in each arm must each sum to zero."""
        rng = np.random.default_rng(3)
        c, t = rng.normal(0, 1, 30), rng.normal(1, 2, 40)
        u = _influence_values(c, t)
        assert u[:30].sum() == pytest.approx(0.0, abs=1e-9)
        assert u[30:].sum() == pytest.approx(0.0, abs=1e-9)

    def test_symmetric_data_gives_near_zero_acceleration(self):
        rng = np.random.default_rng(4)
        c, t = rng.normal(0, 1, 4000), rng.normal(0, 1, 4000)
        r = bootstrap_mean_difference(c, t, n_resamples=500, seed=0)
        assert abs(r.acceleration) < 0.02

    def test_skewed_data_gives_nonzero_acceleration(self):
        rng = np.random.default_rng(5)
        c, t = rng.lognormal(0, 2, 2000), rng.lognormal(0, 2, 2000)
        r = bootstrap_mean_difference(c, t, n_resamples=500, seed=0)
        assert abs(r.acceleration) > 0.01


class TestBasics:
    def test_point_estimate_is_the_observed_difference(self):
        c = np.array([1.0, 2.0, 3.0, 4.0])
        t = np.array([3.0, 4.0, 5.0, 6.0])
        r = bootstrap_mean_difference(c, t, n_resamples=500, seed=0)
        assert r.point == pytest.approx(2.0)

    def test_reproducible_under_the_same_seed(self):
        rng = np.random.default_rng(7)
        c, t = rng.normal(0, 1, 200), rng.normal(0.5, 1, 200)
        a = bootstrap_mean_difference(c, t, n_resamples=1000, seed=42)
        b = bootstrap_mean_difference(c, t, n_resamples=1000, seed=42)
        assert a.ci == b.ci
        assert a.se == b.se

    def test_different_seeds_give_different_intervals(self):
        rng = np.random.default_rng(8)
        c, t = rng.normal(0, 1, 200), rng.normal(0.5, 1, 200)
        a = bootstrap_mean_difference(c, t, n_resamples=1000, seed=1)
        b = bootstrap_mean_difference(c, t, n_resamples=1000, seed=2)
        assert a.ci != b.ci

    def test_rng_and_seed_are_mutually_exclusive(self):
        c, t = np.arange(10.0), np.arange(10.0) + 1
        with pytest.raises(ValueError, match="either seed or rng"):
            bootstrap_mean_difference(c, t, seed=1, rng=np.random.default_rng(1), n_resamples=200)

    def test_accepts_an_explicit_generator(self):
        c, t = np.arange(20.0), np.arange(20.0) + 2
        r = bootstrap_mean_difference(c, t, n_resamples=300, rng=np.random.default_rng(9))
        assert r.point == pytest.approx(2.0)

    def test_too_few_resamples_rejected(self):
        c, t = np.arange(10.0), np.arange(10.0) + 1
        with pytest.raises(ValueError, match="n_resamples must be >= 100"):
            bootstrap_mean_difference(c, t, n_resamples=10)

    def test_single_observation_arm_raises(self):
        with pytest.raises(InsufficientData, match=">= 2 observations"):
            bootstrap_mean_difference(np.array([1.0]), np.arange(10.0), n_resamples=200)

    def test_constant_arms_give_degenerate_interval(self):
        r = bootstrap_mean_difference(np.ones(50), np.full(50, 3.0), n_resamples=200, seed=0)
        assert r.se == 0.0
        assert r.ci == (2.0, 2.0)

    def test_se_shrinks_with_n(self):
        rng = np.random.default_rng(10)
        small = bootstrap_mean_difference(
            rng.normal(0, 1, 100), rng.normal(0, 1, 100), n_resamples=800, seed=0
        )
        large = bootstrap_mean_difference(
            rng.normal(0, 1, 2000), rng.normal(0, 1, 2000), n_resamples=800, seed=0
        )
        assert large.se < small.se


class TestAgreementWithAnalytic:
    def test_percentile_interval_matches_welch_on_normal_data(self):
        """With symmetric data and large n, bootstrap and t-interval should agree."""
        from gatekeeper.frequentist.means import welch_test

        rng = np.random.default_rng(11)
        c, t = rng.normal(10, 3, 4000), rng.normal(10.5, 3, 4000)
        boot = bootstrap_mean_difference(c, t, n_resamples=4000, method="percentile", seed=0)
        analytic = welch_test(c, t)
        assert boot.ci[0] == pytest.approx(analytic.ci[0], abs=0.05)
        assert boot.ci[1] == pytest.approx(analytic.ci[1], abs=0.05)

    def test_bootstrap_se_matches_analytic_se(self):
        from gatekeeper.frequentist.means import welch_test

        rng = np.random.default_rng(12)
        c, t = rng.normal(0, 2, 3000), rng.normal(0.3, 2, 3000)
        boot = bootstrap_mean_difference(c, t, n_resamples=3000, seed=0)
        assert boot.se == pytest.approx(welch_test(c, t).se, rel=0.06)

    def test_bootstrap_on_binary_matches_the_z_test(self):
        """A proportion is a mean, so the two approaches must broadly agree."""
        from gatekeeper.frequentist.proportions import two_proportion_test

        rng = np.random.default_rng(13)
        c = (rng.random(6000) < 0.20).astype(float)
        t = (rng.random(6000) < 0.25).astype(float)
        boot = bootstrap_mean_difference(c, t, n_resamples=3000, seed=0)
        z = two_proportion_test(int(c.sum()), c.size, int(t.sum()), t.size)
        assert boot.point == pytest.approx(z.point)
        assert boot.ci[0] == pytest.approx(z.ci[0], abs=0.008)
        assert boot.ci[1] == pytest.approx(z.ci[1], abs=0.008)


class TestBcaVsPercentile:
    def test_methods_agree_on_symmetric_data(self):
        rng = np.random.default_rng(14)
        c, t = rng.normal(0, 1, 3000), rng.normal(0.2, 1, 3000)
        bca = bootstrap_mean_difference(c, t, n_resamples=3000, method="bca", seed=0)
        pct = bootstrap_mean_difference(c, t, n_resamples=3000, method="percentile", seed=0)
        assert bca.ci[0] == pytest.approx(pct.ci[0], abs=0.02)

    def test_methods_diverge_on_skewed_data(self):
        """If BCa never differs from percentile, its corrections are not being applied."""
        rng = np.random.default_rng(15)
        c, t = rng.lognormal(0, 2.2, 2000), rng.lognormal(0.1, 2.2, 2000)
        bca = bootstrap_mean_difference(c, t, n_resamples=4000, method="bca", seed=0)
        pct = bootstrap_mean_difference(c, t, n_resamples=4000, method="percentile", seed=0)
        assert bca.ci != pct.ci

    def test_method_is_reported_accurately(self):
        rng = np.random.default_rng(16)
        c, t = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
        assert bootstrap_mean_difference(c, t, n_resamples=500, seed=0).method == "bca"
        assert (
            bootstrap_mean_difference(c, t, n_resamples=500, method="percentile", seed=0).method
            == "percentile"
        )


class TestGeneralStatistic:
    def test_median_difference(self):
        rng = np.random.default_rng(17)
        c = rng.lognormal(1.0, 1.0, 500)
        t = rng.lognormal(1.4, 1.0, 500)
        r = bootstrap_statistic(c, t, np.median, n_resamples=600, seed=0)
        expected = float(np.median(t) - np.median(c))
        assert r.point == pytest.approx(expected)
        assert r.ci[0] <= r.point <= r.ci[1]
        assert r.method == "percentile"

    def test_median_bootstrap_detects_a_real_shift(self):
        rng = np.random.default_rng(18)
        c = rng.lognormal(1.0, 0.5, 800)
        t = rng.lognormal(1.8, 0.5, 800)
        r = bootstrap_statistic(c, t, np.median, n_resamples=600, seed=0)
        assert r.ci[0] > 0

    def test_validation(self):
        c, t = np.arange(10.0), np.arange(10.0)
        with pytest.raises(ValueError, match="n_resamples"):
            bootstrap_statistic(c, t, np.median, n_resamples=5)
        with pytest.raises(ValueError, match="either seed or rng"):
            bootstrap_statistic(
                c, t, np.median, n_resamples=200, seed=1, rng=np.random.default_rng(1)
            )
        with pytest.raises(InsufficientData):
            bootstrap_statistic(np.array([1.0]), t, np.median, n_resamples=200)


class TestChunking:
    def test_chunked_resampling_is_correct_for_large_n(self):
        """Peak memory is bounded by chunking; the result must be unaffected."""
        from gatekeeper.frequentist import bootstrap as bs

        rng = np.random.default_rng(19)
        values = rng.normal(0, 1, 30_000)
        original = bs._MAX_RESAMPLE_CELLS
        try:
            # Force many small chunks, then one big one, and compare distributions.
            bs._MAX_RESAMPLE_CELLS = 60_000  # chunk = 2
            small_chunks = bs._resample_means(values, 200, np.random.default_rng(0))
            bs._MAX_RESAMPLE_CELLS = 100_000_000  # single chunk
            one_chunk = bs._resample_means(values, 200, np.random.default_rng(0))
        finally:
            bs._MAX_RESAMPLE_CELLS = original
        # Same generator seed and same draw order => identical results.
        np.testing.assert_allclose(small_chunks, one_chunk, rtol=1e-12)


class TestEstimateWrapper:
    def test_returns_populated_estimate(self):
        exp = make_cookie_cats_like(n=5_000, seed=20)
        est = estimate_bootstrap(
            exp.data,
            Estimand(outcome="sum_gamerounds", treatment="version"),
            n_resamples=800,
            seed=3,
        )
        assert est.method == "bootstrap_bca"
        assert est.seed == 3
        assert est.assumptions
        assert "acceleration" in est.diagnostics
        assert any("resamples" in a for a in est.assumptions)

    def test_relative_scale_is_refused_rather_than_faked(self):
        exp = make_cookie_cats_like(n=1_000, seed=21)
        with pytest.raises(NotImplementedError, match="relative-scale bootstrap"):
            estimate_bootstrap(
                exp.data,
                Estimand(outcome="sum_gamerounds", treatment="version", scale=Scale.RELATIVE),
                n_resamples=200,
            )

    def test_percentile_method_is_recorded_in_the_method_name(self):
        exp = make_cookie_cats_like(n=2_000, seed=22)
        est = estimate_bootstrap(
            exp.data,
            Estimand(outcome="sum_gamerounds", treatment="version"),
            n_resamples=500,
            method="percentile",
        )
        assert est.method == "bootstrap_percentile"
