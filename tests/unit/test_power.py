"""Power, sample size, MDE, and duration."""

from __future__ import annotations

import numpy as np
import pytest

from gatekeeper.design.power import (
    duration_days,
    mde_means,
    mde_two_proportion,
    power_means,
    power_two_proportion,
    sample_size_means,
    sample_size_two_proportion,
)
from gatekeeper.frequentist.proportions import two_proportion_test
from gatekeeper.types import InsufficientData


class TestPowerFundamentals:
    def test_power_at_the_null_equals_alpha(self):
        """The keystone identity: with no effect, rejection happens at rate alpha.

        This is what the pooled/unpooled asymmetry in power_two_proportion buys, and
        it fails if either variance is used for both roles.
        """
        for alpha in (0.01, 0.05, 0.10):
            p = power_two_proportion(0.2, 0.0, 10_000, alpha=alpha)
            assert p == pytest.approx(alpha, abs=1e-9)

    def test_power_at_the_null_equals_alpha_for_means(self):
        for alpha in (0.01, 0.05, 0.10):
            p = power_means(0.0, 1.0, 5_000, alpha=alpha)
            assert p == pytest.approx(alpha, abs=1e-9)

    def test_power_increases_with_sample_size(self):
        powers = [power_two_proportion(0.2, 0.02, n) for n in (500, 1_000, 5_000, 20_000)]
        assert powers == sorted(powers)
        assert powers[-1] > 0.99

    def test_power_increases_with_effect_size(self):
        powers = [power_two_proportion(0.2, e, 2_000) for e in (0.005, 0.01, 0.03, 0.08)]
        assert powers == sorted(powers)

    def test_power_is_symmetric_in_sign_for_means(self):
        assert power_means(0.1, 1.0, 500) == pytest.approx(power_means(-0.1, 1.0, 500))

    def test_nct_and_normal_branches_agree_across_the_boundary(self):
        """The df threshold must not introduce a visible discontinuity.

        Below the limit power_means uses the exact non-central t; above it, the normal
        approximation (scipy's nct returns NaN at large df). Straddling the boundary
        must not change the answer materially.
        """
        from gatekeeper.design.power import _NCT_DF_LIMIT

        # n per arm either side of df = n_c + n_t - 2 = _NCT_DF_LIMIT
        n_below = int(_NCT_DF_LIMIT // 2)  # df just under the limit -> nct branch
        n_above = n_below + 2  # df just over -> normal branch
        for effect in (0.05, 0.1, 0.2):
            below = power_means(effect, 1.0, n_below)
            above = power_means(effect, 1.0, n_above)
            assert below == pytest.approx(above, abs=2e-3), (
                f"branch discontinuity at effect={effect}: {below} vs {above}"
            )

    def test_large_n_does_not_return_nan(self):
        """Regression: scipy's nct produced NaN and broke the sample-size solver."""
        for n in (5_000, 100_000, 10_000_000):
            p = power_means(0.01, 1.0, n)
            assert 0.0 <= p <= 1.0
            assert p == p  # not NaN

    def test_power_is_not_symmetric_in_sign_for_proportions(self):
        """Binomial variance depends on p, so up and down are not equally detectable."""
        up = power_two_proportion(0.05, 0.02, 2_000)
        down = power_two_proportion(0.05, -0.02, 2_000)
        assert up != pytest.approx(down, rel=1e-6)

    def test_impossible_effect_raises(self):
        with pytest.raises(ValueError, match="outside"):
            power_two_proportion(0.9, 0.2, 1_000)

    def test_bad_inputs_raise(self):
        with pytest.raises(ValueError, match="alpha"):
            power_two_proportion(0.2, 0.01, 100, alpha=1.0)
        with pytest.raises(ValueError, match="p_control"):
            power_two_proportion(0.0, 0.01, 100)
        with pytest.raises(InsufficientData, match=">= 1 unit"):
            power_two_proportion(0.2, 0.01, 0)
        with pytest.raises(ValueError, match="sd must be positive"):
            power_means(0.1, 0.0, 100)


class TestRoundTrip:
    """sample_size and mde invert power, so composing them must be the identity."""

    @pytest.mark.parametrize("p_control", [0.05, 0.19, 0.5])
    @pytest.mark.parametrize("effect", [0.01, 0.02, 0.05])
    def test_sample_size_then_power_meets_target(self, p_control: float, effect: float):
        n = sample_size_two_proportion(p_control, effect, power=0.80)
        assert power_two_proportion(p_control, effect, n) >= 0.80
        # And it is the SMALLEST such n.
        assert power_two_proportion(p_control, effect, n - 1) < 0.80

    @pytest.mark.parametrize("n", [1_000, 10_000, 45_000])
    def test_mde_then_power_hits_target(self, n: int):
        mde = mde_two_proportion(0.19, n, power=0.80)
        assert power_two_proportion(0.19, mde, n) == pytest.approx(0.80, abs=1e-6)

    def test_sample_size_and_mde_are_mutually_consistent(self):
        n = sample_size_two_proportion(0.19, 0.0075, power=0.80)
        mde = mde_two_proportion(0.19, n, power=0.80)
        # The MDE at the required n should be at or just under the effect asked for.
        assert mde <= 0.0075 + 1e-9

    def test_means_round_trip(self):
        n = sample_size_means(0.5, 2.0, power=0.80)
        assert power_means(0.5, 2.0, n) >= 0.80
        assert power_means(0.5, 2.0, n - 1) < 0.80

    def test_means_mde_round_trip(self):
        mde = mde_means(2.0, 1_000, power=0.80)
        assert power_means(mde, 2.0, 1_000) == pytest.approx(0.80, abs=1e-6)

    def test_mde_decrease_direction(self):
        down = mde_two_proportion(0.19, 10_000, direction="decrease")
        assert down < 0
        assert power_two_proportion(0.19, down, 10_000) == pytest.approx(0.80, abs=1e-6)

    def test_mde_directions_are_not_mirror_images(self):
        up = mde_two_proportion(0.05, 5_000, direction="increase")
        down = mde_two_proportion(0.05, 5_000, direction="decrease")
        assert abs(up) != pytest.approx(abs(down), rel=1e-6)

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            mde_two_proportion(0.2, 1_000, direction="sideways")

    def test_zero_effect_cannot_be_powered(self):
        with pytest.raises(ValueError, match="exactly zero"):
            sample_size_two_proportion(0.2, 0.0)

    def test_power_below_alpha_is_rejected(self):
        with pytest.raises(ValueError, match="must exceed alpha"):
            sample_size_two_proportion(0.2, 0.01, alpha=0.05, power=0.04)

    def test_unreachable_mde_raises(self):
        with pytest.raises(InsufficientData, match="cannot answer this question"):
            mde_two_proportion(0.5, 3, power=0.99)


class TestSpecConsistency:
    """The committed spec's mde must actually match what power.py computes."""

    def test_committed_spec_mde_is_honest(self):
        from gatekeeper.data.ingest import project_root
        from gatekeeper.spec import load_spec

        spec = load_spec(project_root() / "specs" / "cookie_cats_gate.yaml")
        # The spec's mde is stated for a ~19% base rate at ~45k units per arm.
        computed = mde_two_proportion(0.19, 45_000, alpha=spec.alpha, power=spec.power)
        assert computed == pytest.approx(spec.mde, abs=0.0005), (
            f"spec declares mde={spec.mde} but power.py computes {computed:.6f}; "
            "the spec is wrong and must be corrected before Phase 3"
        )

    def test_spec_mde_clears_its_practical_threshold(self):
        from gatekeeper.data.ingest import project_root
        from gatekeeper.spec import load_spec

        spec = load_spec(project_root() / "specs" / "cookie_cats_gate.yaml")
        assert spec.mde <= spec.practical_threshold


class TestPowerAgainstSimulation:
    """Validate analytic power empirically: the rejection rate must match.

    A stronger check than matching another library's parameterisation, and it is the
    same discipline the calibration suite applies to the estimators.
    """

    @pytest.mark.parametrize(
        ("p_control", "effect", "n"),
        [(0.20, 0.05, 400), (0.50, 0.06, 500), (0.10, 0.04, 800)],
    )
    def test_analytic_power_matches_empirical_rejection_rate(
        self, p_control: float, effect: float, n: int
    ):
        analytic = power_two_proportion(p_control, effect, n)
        rng = np.random.default_rng(20260817)
        n_sims = 4_000
        rejections = 0
        for _ in range(n_sims):
            s_c = int(rng.binomial(n, p_control))
            s_t = int(rng.binomial(n, p_control + effect))
            r = two_proportion_test(s_c, n, s_t, n, warn_small=False)
            rejections += r.p_value < 0.05
        empirical = rejections / n_sims
        # Monte Carlo SE at p~0.5 with 4000 sims is ~0.008; allow ~3.5 SE.
        assert empirical == pytest.approx(analytic, abs=0.03), (
            f"analytic power {analytic:.4f} vs empirical {empirical:.4f}"
        )


class TestDuration:
    def test_hand_computed(self):
        assert duration_days(1_000, 2, 500.0) == pytest.approx(4.0)

    def test_more_traffic_is_faster(self):
        assert duration_days(1_000, 2, 1_000.0) < duration_days(1_000, 2, 100.0)

    def test_more_arms_take_longer(self):
        assert duration_days(1_000, 3, 500.0) > duration_days(1_000, 2, 500.0)

    def test_validation(self):
        with pytest.raises(ValueError, match="n_per_arm"):
            duration_days(0, 2, 100.0)
        with pytest.raises(ValueError, match="n_arms"):
            duration_days(100, 1, 100.0)
        with pytest.raises(ValueError, match="units_per_day"):
            duration_days(100, 2, 0.0)
