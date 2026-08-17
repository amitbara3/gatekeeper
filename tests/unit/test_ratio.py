"""Delta-method ratio inference.

The headline test here is :meth:`TestCovarianceTerm.test_perfectly_correlated_gives_zero_variance`.
It is the cleanest possible demonstration that the covariance term is required: when the
numerator equals the denominator the ratio is exactly 1 with no sampling variability, and
only the full formula returns zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from gatekeeper.frequentist.ratio import ratio_difference, ratio_variance
from gatekeeper.types import InsufficientData


class TestKnownAnswers:
    def test_ratio_of_constant_arrays(self):
        num = np.full(20, 6.0)
        den = np.full(20, 3.0)
        r = ratio_variance(num, den)
        assert r.ratio == pytest.approx(2.0)
        assert r.variance == pytest.approx(0.0)

    def test_ratio_is_a_ratio_of_means_not_a_mean_of_ratios(self):
        """These differ, and the ratio-of-means is the right one for a rate metric."""
        num = np.array([0.0, 10.0])
        den = np.array([1.0, 10.0])
        r = ratio_variance(num, den)
        assert r.ratio == pytest.approx(5.0 / 5.5)
        mean_of_ratios = float(np.mean(num / den))  # = 0.5
        assert r.ratio != pytest.approx(mean_of_ratios)


class TestCovarianceTerm:
    def test_perfectly_correlated_gives_zero_variance(self):
        """Y == X exactly: the ratio is identically 1, so its variance is exactly 0.

        var(Y) - 2R*cov(Y,X) + R^2*var(X)  with Y=X, R=1
          = v - 2v + v = 0

        Drop the covariance term and you would get 2v/(n*xbar^2) instead -- a positive
        variance for a quantity that cannot vary. Conservative, and wrong.
        """
        rng = np.random.default_rng(0)
        x = rng.lognormal(1.0, 0.8, 500)
        r = ratio_variance(x.copy(), x.copy())
        assert r.ratio == pytest.approx(1.0)
        assert r.variance == pytest.approx(0.0, abs=1e-18)

    def test_ignoring_covariance_would_overstate_variance(self):
        rng = np.random.default_rng(1)
        den = rng.poisson(8, 2000).astype(float) + 1.0
        num = den * 0.3 + rng.normal(0, 0.4, 2000)  # strongly correlated with den
        r = ratio_variance(num, den)

        n = den.size
        mean_x = den.mean()
        naive = (num.var(ddof=1) + r.ratio**2 * den.var(ddof=1)) / (n * mean_x**2)
        assert r.variance < naive, "covariance term must reduce the variance here"

    def test_independent_numerator_and_denominator(self):
        """With zero covariance the formula reduces to the naive sum."""
        rng = np.random.default_rng(2)
        num = rng.normal(10, 2, 4000)
        den = rng.normal(5, 1, 4000)
        r = ratio_variance(num, den)
        n = den.size
        expected = (num.var(ddof=1) + r.ratio**2 * den.var(ddof=1)) / (n * den.mean() ** 2)
        # Sample covariance is near zero but not exactly, so allow a little slack.
        assert r.variance == pytest.approx(expected, rel=0.10)


class TestValidation:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="align elementwise"):
            ratio_variance(np.arange(5.0), np.arange(6.0))

    def test_single_unit_raises(self):
        with pytest.raises(InsufficientData, match=">= 2 units"):
            ratio_variance(np.array([1.0]), np.array([1.0]))

    def test_zero_denominator_mean_raises(self):
        with pytest.raises(InsufficientData, match="denominator mean is zero"):
            ratio_variance(np.arange(5.0), np.zeros(5))

    def test_negative_denominator_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ratio_variance(np.arange(5.0), np.array([1.0, 1.0, -1.0, 1.0, 1.0]))

    def test_variance_is_never_negative(self):
        """Taylor truncation can push a near-zero variance slightly negative."""
        rng = np.random.default_rng(3)
        for _ in range(20):
            x = rng.lognormal(0, 1, 30)
            assert ratio_variance(x.copy(), x.copy()).variance >= 0.0

    def test_bad_alpha_raises(self):
        a = np.arange(1.0, 11.0)
        with pytest.raises(ValueError, match="alpha"):
            ratio_difference(a, a, a, a, alpha=0.0)


class TestRatioDifference:
    def test_identical_arms_give_zero_difference(self):
        rng = np.random.default_rng(4)
        den = rng.poisson(5, 400).astype(float) + 1
        num = den * 0.4
        r = ratio_difference(num, den, num.copy(), den.copy())
        assert r.point == pytest.approx(0.0)
        assert r.p_value == pytest.approx(1.0)

    def test_detects_a_real_difference(self):
        rng = np.random.default_rng(5)
        den_c = rng.poisson(10, 3000).astype(float) + 1
        den_t = rng.poisson(10, 3000).astype(float) + 1
        num_c = den_c * 0.30 + rng.normal(0, 0.5, 3000)
        num_t = den_t * 0.40 + rng.normal(0, 0.5, 3000)
        r = ratio_difference(num_c, den_c, num_t, den_t)
        assert r.point == pytest.approx(0.10, abs=0.02)
        assert r.ci[0] > 0
        assert r.p_value < 0.001

    def test_swapping_arms_flips_the_sign(self):
        rng = np.random.default_rng(6)
        den_c = rng.poisson(6, 500).astype(float) + 1
        den_t = rng.poisson(6, 500).astype(float) + 1
        num_c = den_c * 0.2 + rng.normal(0, 0.3, 500)
        num_t = den_t * 0.3 + rng.normal(0, 0.3, 500)
        fwd = ratio_difference(num_c, den_c, num_t, den_t)
        rev = ratio_difference(num_t, den_t, num_c, den_c)
        assert fwd.point == pytest.approx(-rev.point)
        assert fwd.p_value == pytest.approx(rev.p_value)
        assert fwd.se == pytest.approx(rev.se)

    def test_point_lies_inside_interval(self):
        rng = np.random.default_rng(7)
        den = rng.poisson(4, 300).astype(float) + 1
        num = den * 0.5 + rng.normal(0, 0.2, 300)
        r = ratio_difference(num, den, num * 1.1, den)
        assert r.ci[0] <= r.point <= r.ci[1]

    def test_arm_variances_add(self):
        rng = np.random.default_rng(8)
        den = rng.poisson(7, 600).astype(float) + 1
        num = den * 0.3 + rng.normal(0, 0.4, 600)
        r = ratio_difference(num, den, num.copy(), den.copy())
        expected = np.sqrt(r.control.variance + r.treatment.variance)
        assert r.se == pytest.approx(expected)

    def test_se_property_on_arm_estimate(self):
        rng = np.random.default_rng(9)
        den = rng.poisson(5, 200).astype(float) + 1
        num = den * 0.4 + rng.normal(0, 0.3, 200)
        est = ratio_variance(num, den)
        assert est.se == pytest.approx(np.sqrt(est.variance))
        assert est.n == 200
