"""Cross-check our estimators against scipy and statsmodels.

Architecture §6 layer 2. A hand-written fixture proves one case; a property test proves
an invariant; neither catches a systematic algebra error that happens to be
self-consistent. Agreement with an independent implementation across the input space
does.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportions_ztest

from gatekeeper.design.power import power_means
from gatekeeper.frequentist.means import welch_test
from gatekeeper.frequentist.multiplicity import correct
from gatekeeper.frequentist.proportions import two_proportion_test

TOL = 1e-9


class TestProportionsVsStatsmodels:
    """statsmodels' proportions_ztest uses the pooled variance, as our p-value does."""

    @pytest.mark.parametrize(
        ("s_c", "n_c", "s_t", "n_t"),
        [
            (200, 1000, 250, 1000),
            (20, 100, 30, 100),
            (8_550, 45_000, 8_190, 45_000),
            (1, 50, 40, 50),
            (500, 1000, 500, 1000),
            (100, 1000, 900, 5000),
        ],
    )
    def test_p_value_matches(self, s_c: int, n_c: int, s_t: int, n_t: int):
        ours = two_proportion_test(s_c, n_c, s_t, n_t, warn_small=False)
        # statsmodels takes (count, nobs) arrays; order is [control, treatment].
        _, their_p = proportions_ztest(count=np.array([s_t, s_c]), nobs=np.array([n_t, n_c]))
        assert ours.p_value == pytest.approx(float(their_p), rel=TOL, abs=TOL)

    @pytest.mark.parametrize(
        ("s_c", "n_c", "s_t", "n_t"),
        [(200, 1000, 250, 1000), (30, 200, 55, 210), (8_550, 45_000, 8_190, 45_000)],
    )
    def test_z_statistic_matches_in_magnitude(self, s_c: int, n_c: int, s_t: int, n_t: int):
        ours = two_proportion_test(s_c, n_c, s_t, n_t, warn_small=False)
        their_z, _ = proportions_ztest(count=np.array([s_t, s_c]), nobs=np.array([n_t, n_c]))
        assert abs(ours.z) == pytest.approx(abs(float(their_z)), rel=TOL, abs=TOL)

    @settings(max_examples=200, deadline=None)
    @given(
        s_c=st.integers(min_value=1, max_value=4_999),
        s_t=st.integers(min_value=1, max_value=4_999),
    )
    def test_agreement_across_the_input_space(self, s_c: int, s_t: int):
        n = 5_000
        ours = two_proportion_test(s_c, n, s_t, n, warn_small=False)
        _, their_p = proportions_ztest(count=np.array([s_t, s_c]), nobs=np.array([n, n]))
        assert ours.p_value == pytest.approx(float(their_p), rel=1e-9, abs=1e-12)


class TestWelchVsScipy:
    @pytest.mark.parametrize("seed", range(6))
    def test_matches_ttest_ind_unequal_var(self, seed: int):
        rng = np.random.default_rng(seed)
        c = rng.normal(10, 2 + seed, 50 + seed * 10)
        t = rng.normal(11, 5 - seed * 0.5, 60 + seed * 5)

        ours = welch_test(c, t)
        their = stats.ttest_ind(t, c, equal_var=False)

        assert ours.t == pytest.approx(float(their.statistic), rel=TOL, abs=TOL)
        assert ours.p_value == pytest.approx(float(their.pvalue), rel=TOL, abs=TOL)
        assert ours.df == pytest.approx(float(their.df), rel=TOL, abs=TOL)

    @pytest.mark.parametrize("seed", range(4))
    def test_confidence_interval_matches_scipy(self, seed: int):
        rng = np.random.default_rng(100 + seed)
        c = rng.normal(0, 1 + seed, 80)
        t = rng.normal(0.5, 3, 90)

        ours = welch_test(c, t, alpha=0.05)
        their = stats.ttest_ind(t, c, equal_var=False).confidence_interval(0.95)
        assert ours.ci[0] == pytest.approx(float(their.low), rel=1e-9, abs=1e-9)
        assert ours.ci[1] == pytest.approx(float(their.high), rel=1e-9, abs=1e-9)

    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        loc=st.floats(min_value=-20, max_value=20, allow_nan=False),
        sd_ratio=st.floats(min_value=0.2, max_value=5.0, allow_nan=False),
    )
    def test_agreement_across_the_input_space(self, loc: float, sd_ratio: float):
        rng = np.random.default_rng(7)
        c = rng.normal(0, 1, 60)
        t = rng.normal(loc, sd_ratio, 70)
        ours = welch_test(c, t)
        their = stats.ttest_ind(t, c, equal_var=False)
        assert ours.p_value == pytest.approx(float(their.pvalue), rel=1e-9, abs=1e-12)


class TestPowerVsStatsmodels:
    """Validates the exact (non-central t) branch of power_means.

    statsmodels parameterises by Cohen's d = effect / sd, assuming equal variance --
    the same assumption power_means documents for its planning-time model.
    """

    @pytest.mark.parametrize("n", [20, 50, 120, 400])
    @pytest.mark.parametrize("d", [0.1, 0.25, 0.5, 0.8])
    def test_matches_ttestindpower(self, n: int, d: float):
        ours = power_means(effect=d, sd=1.0, n_control=n, alpha=0.05)
        theirs = float(TTestIndPower().power(effect_size=d, nobs1=n, ratio=1.0, alpha=0.05))

        if not math.isfinite(theirs):
            # On some scipy versions nct.cdf(-crit, df, nc) returns NaN (observed at
            # d=0.8, n=400: df=798, nc=11.3 under scipy 1.17.1) and statsmodels
            # propagates it. Our isfinite guard falls back to the normal approximation,
            # so we stay well-defined where the reference does not.
            #
            # Whether this branch is reached is a *version* detail, so nothing here
            # asserts that statsmodels misbehaves -- only that we do not. An earlier
            # version of this test asserted the NaN and failed in CI, where scipy
            # returns a finite value.
            assert math.isfinite(ours), "we must never propagate scipy's nct NaN"
            assert ours == pytest.approx(1.0, abs=1e-6), (
                "NaN from nct arises in the saturated-power regime, so the correct "
                f"answer is ~1.0; got {ours}"
            )
            return

        assert ours == pytest.approx(theirs, rel=1e-7, abs=1e-9)

    def test_power_is_always_finite_across_a_wide_grid(self):
        """Our own invariant, asserted without reference to any other library.

        Power is a probability: it is always defined and always in [0, 1]. Upholding
        that regardless of how the backend behaves is the entire job of the
        ``_NCT_DF_LIMIT`` branch and the ``isfinite`` fallback in ``power_means``.

        This replaces a test that asserted statsmodels returns NaN at a specific input.
        That assertion held locally and failed in CI, because it encoded a property of
        one scipy build rather than a property of our code. A test should pin down what
        we guarantee, not what a dependency happens to get wrong this week.
        """
        for d in (0.001, 0.01, 0.1, 0.5, 0.8, 2.0, 10.0):
            for n in (2, 5, 20, 400, 5_000, 100_000, 10**7):
                p = power_means(effect=d, sd=1.0, n_control=n, alpha=0.05)
                assert math.isfinite(p), f"non-finite power at d={d}, n={n}: {p}"
                assert 0.0 <= p <= 1.0, f"power out of range at d={d}, n={n}: {p}"

    def test_matches_with_unequal_group_sizes(self):
        ours = power_means(effect=0.4, sd=1.0, n_control=60, n_treatment=120, alpha=0.05)
        theirs = TTestIndPower().power(effect_size=0.4, nobs1=60, ratio=2.0, alpha=0.05)
        assert ours == pytest.approx(float(theirs), rel=1e-7, abs=1e-9)

    def test_scaling_sd_is_equivalent_to_scaling_effect(self):
        a = power_means(effect=1.0, sd=4.0, n_control=100)
        b = power_means(effect=0.25, sd=1.0, n_control=100)
        assert a == pytest.approx(b, rel=1e-12)


class TestMultiplicityVsStatsmodels:
    @pytest.mark.parametrize(
        "p_values",
        [
            [0.01, 0.02, 0.03, 0.04, 0.05],
            [0.001, 0.5],
            [0.04, 0.01],
            [0.02, 0.02, 0.02],
            [0.0001, 0.01, 0.03, 0.2, 0.5, 0.9],
            [0.5],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
    )
    def test_bh_matches(self, p_values: list[float]):
        ours = correct(p_values, alpha=0.05, method="benjamini_hochberg")
        rejected, adjusted, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        np.testing.assert_allclose(ours.adjusted, adjusted, rtol=1e-12, atol=1e-15)
        assert list(ours.rejected) == list(rejected)

    @pytest.mark.parametrize("p_values", [[0.01, 0.02, 0.03], [0.001, 0.5], [0.4, 0.6, 0.8]])
    def test_bonferroni_matches(self, p_values: list[float]):
        ours = correct(p_values, alpha=0.05, method="bonferroni")
        rejected, adjusted, _, _ = multipletests(p_values, alpha=0.05, method="bonferroni")
        np.testing.assert_allclose(ours.adjusted, adjusted, rtol=1e-12, atol=1e-15)
        assert list(ours.rejected) == list(rejected)

    @settings(max_examples=200, deadline=None)
    @given(
        p_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=25,
        )
    )
    def test_bh_agreement_across_the_input_space(self, p_values: list[float]):
        ours = correct(p_values, alpha=0.05)
        _, adjusted, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        np.testing.assert_allclose(ours.adjusted, adjusted, rtol=1e-10, atol=1e-13)
