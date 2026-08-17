"""Always-valid p-values, and the peeking problem they solve."""

from __future__ import annotations

import numpy as np
import pytest

from gatekeeper.sequential.always_valid import (
    always_valid_interval,
    always_valid_p_value,
    sequential_p_values,
    suggest_tau,
)
from gatekeeper.types import InsufficientData


class TestAlwaysValidPValue:
    def test_zero_effect_gives_a_p_value_of_one(self):
        assert always_valid_p_value(0.0, 0.01, tau=0.05) == pytest.approx(1.0)

    def test_large_effect_gives_a_small_p_value(self):
        p = always_valid_p_value(0.20, 0.0001, tau=0.05)
        assert p < 0.001

    def test_p_value_is_always_in_the_unit_interval(self):
        for estimate in (-1.0, -0.1, 0.0, 0.001, 0.1, 1.0, 100.0):
            for variance in (1e-8, 1e-4, 1.0, 100.0):
                p = always_valid_p_value(estimate, variance, tau=0.05)
                assert 0.0 <= p <= 1.0, f"p={p} at estimate={estimate}, var={variance}"

    def test_symmetric_in_the_sign_of_the_effect(self):
        assert always_valid_p_value(0.05, 0.001, tau=0.05) == pytest.approx(
            always_valid_p_value(-0.05, 0.001, tau=0.05)
        )

    def test_p_value_decreases_as_variance_shrinks(self):
        ps = [always_valid_p_value(0.05, v, tau=0.05) for v in (1e-2, 1e-3, 1e-4, 1e-5)]
        assert ps == sorted(ps, reverse=True)

    def test_no_overflow_at_extreme_effects(self):
        """Lambda overflows a float outright at large effects; we work in logs."""
        p = always_valid_p_value(1e4, 1e-8, tau=1.0)
        assert p == pytest.approx(0.0, abs=1e-300)
        assert not np.isnan(p)

    def test_always_valid_is_more_conservative_than_fixed_horizon(self):
        """The price of unrestricted monitoring, made explicit."""
        from scipy import stats

        estimate, variance = 0.02, 0.0001
        se = np.sqrt(variance)
        fixed = float(2 * stats.norm.sf(abs(estimate / se)))
        sequential = always_valid_p_value(estimate, variance, tau=0.02)
        assert sequential > fixed

    def test_validation(self):
        with pytest.raises(ValueError, match="tau must be positive"):
            always_valid_p_value(0.1, 0.01, tau=0.0)
        with pytest.raises(InsufficientData, match="variance must be positive"):
            always_valid_p_value(0.1, 0.0, tau=0.05)


class TestSuggestTau:
    def test_uses_the_mde(self):
        assert suggest_tau(0.0075) == pytest.approx(0.0075)

    def test_rejects_non_positive_mde(self):
        with pytest.raises(ValueError, match="mde must be positive"):
            suggest_tau(0.0)


class TestAlwaysValidInterval:
    def test_interval_contains_the_estimate(self):
        lo, hi = always_valid_interval(0.03, 0.0001, tau=0.03)
        assert lo <= 0.03 <= hi

    def test_interval_is_symmetric_about_the_estimate(self):
        estimate = 0.03
        lo, hi = always_valid_interval(estimate, 0.0001, tau=0.03)
        assert (estimate - lo) == pytest.approx(hi - estimate)

    def test_interval_is_wider_than_the_fixed_horizon_one(self):
        from scipy import stats

        estimate, variance = 0.03, 0.0001
        se = np.sqrt(variance)
        fixed_half = float(stats.norm.isf(0.025) * se)
        lo, hi = always_valid_interval(estimate, variance, tau=0.03, alpha=0.05)
        assert (hi - lo) / 2 > fixed_half

    def test_interval_narrows_as_variance_shrinks(self):
        wide = always_valid_interval(0.03, 1e-3, tau=0.03)
        tight = always_valid_interval(0.03, 1e-6, tau=0.03)
        assert (tight[1] - tight[0]) < (wide[1] - wide[0])

    def test_agrees_with_the_p_value_at_the_boundary(self):
        """Inverting the test must reproduce it: at the edge, p == alpha."""
        estimate, variance, tau, alpha = 0.05, 1e-4, 0.05, 0.05
        lo, _ = always_valid_interval(estimate, variance, tau, alpha=alpha)
        # Deviation from the boundary value is exactly the rejection threshold.
        p_at_boundary = always_valid_p_value(estimate - lo, variance, tau)
        assert p_at_boundary == pytest.approx(alpha, rel=1e-6)

    def test_validation(self):
        with pytest.raises(ValueError, match="alpha"):
            always_valid_interval(0.1, 0.01, tau=0.05, alpha=1.5)
        with pytest.raises(InsufficientData, match="variance"):
            always_valid_interval(0.1, -1.0, tau=0.05)


class TestSequentialPath:
    def test_running_minimum_is_monotone(self):
        estimates = np.array([0.10, 0.01, 0.005, 0.02, 0.03])
        variances = np.full(5, 1e-4)
        result = sequential_p_values(estimates, variances, tau=0.05)
        assert list(result.p_values) == sorted(result.p_values, reverse=True)

    def test_running_minimum_remembers_an_earlier_crossing(self):
        """A dip below alpha must not be undone by later data.

        The stopping decision was available at the earlier look, so the guarantee has
        to account for it -- which is what makes monotone p-values necessary.
        """
        estimates = np.array([0.30, 0.001])  # big effect, then nothing
        variances = np.array([1e-5, 1e-5])
        result = sequential_p_values(estimates, variances, tau=0.05, alpha=0.05)
        assert result.stopped_early
        assert result.first_crossing == 0
        assert result.p_values[1] <= result.p_values[0]
        assert result.raw_p_values[1] > result.p_values[1]

    def test_no_crossing_when_there_is_no_effect(self):
        estimates = np.zeros(10)
        variances = np.full(10, 1e-4)
        result = sequential_p_values(estimates, variances, tau=0.05)
        assert not result.stopped_early
        assert result.first_crossing is None
        assert all(p == pytest.approx(1.0) for p in result.p_values)

    def test_records_tau(self):
        result = sequential_p_values(np.zeros(3), np.full(3, 1e-4), tau=0.07)
        assert result.tau == 0.07

    def test_validation(self):
        with pytest.raises(ValueError, match="must align"):
            sequential_p_values(np.zeros(3), np.zeros(4), tau=0.05)
        with pytest.raises(InsufficientData, match="no looks"):
            sequential_p_values(np.array([]), np.array([]), tau=0.05)
