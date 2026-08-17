"""Measure the peeking problem, and verify the fix controls it.

Predictions written down before running (R2.2), from Phases.md Phase 4:

1. Reading a fixed-horizon p-value at every look and stopping at the first ``p < 0.05``
   inflates the false-positive rate far above 5%, and the inflation grows with the
   number of looks.
2. The mSPRT always-valid p-value holds the rate at or below 5% regardless of how many
   looks are taken.
3. Reading once at the end sits at exactly 5%, which confirms the harness itself is
   unbiased rather than the effect being an artefact of the simulation.

All three are asserted below. Marked slow: each case simulates thousands of monitored
experiments.
"""

from __future__ import annotations

import numpy as np
import pytest

from gatekeeper.sequential.always_valid import always_valid_p_value, sequential_p_values
from gatekeeper.sequential.peeking import simulate_peeking

pytestmark = pytest.mark.slow

ALPHA = 0.05
N_SIMS = 1_500


class TestThePeekingProblem:
    def test_reading_once_at_the_end_is_correctly_calibrated(self):
        """Prediction 3: validates the harness before trusting its other outputs."""
        r = simulate_peeking(
            "final_only", n_looks=10, alpha=ALPHA, n_sims=N_SIMS, seed=1, n_per_arm_final=4_000
        )
        se = np.sqrt(ALPHA * (1 - ALPHA) / N_SIMS)
        assert r.rejection_rate == pytest.approx(ALPHA, abs=4 * se), r.describe()

    def test_naive_peeking_inflates_the_error_rate(self):
        """Prediction 1. This number is why R1.5 exists."""
        r = simulate_peeking(
            "naive", n_looks=10, alpha=ALPHA, n_sims=N_SIMS, seed=2, n_per_arm_final=4_000
        )
        assert r.rejection_rate > 2 * ALPHA, (
            f"expected substantial inflation from 10 looks, got {r.describe()}"
        )
        assert not r.controls_error_rate

    def test_inflation_grows_with_the_number_of_looks(self):
        rates = []
        for n_looks in (1, 2, 5, 10, 20):
            r = simulate_peeking(
                "naive",
                n_looks=n_looks,
                alpha=ALPHA,
                n_sims=N_SIMS,
                seed=3,
                n_per_arm_final=4_000,
            )
            rates.append(r.rejection_rate)
        # Monotone up to Monte Carlo noise; the endpoints must be clearly ordered.
        assert rates[0] < rates[-1] / 2, f"rates by look count: {rates}"
        assert rates[0] == pytest.approx(ALPHA, abs=0.025), (
            f"a single look should be calibrated, got {rates[0]:.4f}"
        )

    def test_always_valid_controls_the_error_rate(self):
        """Prediction 2 -- the whole point of the module."""
        r = simulate_peeking(
            "always_valid",
            n_looks=10,
            alpha=ALPHA,
            n_sims=N_SIMS,
            seed=4,
            n_per_arm_final=4_000,
        )
        assert r.controls_error_rate, r.describe()

    def test_always_valid_controls_the_rate_at_many_looks(self):
        """The guarantee must not degrade as monitoring becomes more frequent."""
        for n_looks in (5, 20, 50):
            r = simulate_peeking(
                "always_valid",
                n_looks=n_looks,
                alpha=ALPHA,
                n_sims=800,
                seed=5,
                n_per_arm_final=4_000,
            )
            assert r.controls_error_rate, f"failed at {n_looks} looks: {r.describe()}"

    def test_always_valid_beats_naive_at_the_same_look_count(self):
        naive = simulate_peeking(
            "naive", n_looks=10, alpha=ALPHA, n_sims=N_SIMS, seed=6, n_per_arm_final=4_000
        )
        valid = simulate_peeking(
            "always_valid",
            n_looks=10,
            alpha=ALPHA,
            n_sims=N_SIMS,
            seed=6,
            n_per_arm_final=4_000,
        )
        assert valid.rejection_rate < naive.rejection_rate


class TestAlwaysValidGuarantee:
    """Test the martingale guarantee directly: P(exists n: p_n <= alpha) <= alpha."""

    def test_path_wise_guarantee_under_the_null(self):
        rng = np.random.default_rng(7)
        n_sims, n_looks = 3_000, 25
        tau, sigma = 0.05, 1.0
        crossings = 0
        for _ in range(n_sims):
            # A random walk of cumulative means under a true null effect of zero.
            n_grid = np.linspace(50, 2_000, n_looks).astype(int)
            noise = rng.standard_normal(n_grid[-1])
            cumulative = np.cumsum(noise)
            estimates = cumulative[n_grid - 1] / n_grid
            variances = sigma**2 / n_grid
            if sequential_p_values(estimates, variances, tau, alpha=ALPHA).stopped_early:
                crossings += 1
        rate = crossings / n_sims
        se = np.sqrt(ALPHA * (1 - ALPHA) / n_sims)
        assert rate <= ALPHA + 3 * se, (
            f"always-valid p-value crossed alpha on {rate:.4f} of null paths, "
            f"above the {ALPHA} guarantee"
        )

    def test_guarantee_holds_for_a_range_of_tau(self):
        """tau affects power, not validity. Any positive value must be safe."""
        rng = np.random.default_rng(8)
        n_sims, n_looks = 1_200, 20
        for tau in (0.005, 0.05, 0.5):
            crossings = 0
            for _ in range(n_sims):
                n_grid = np.linspace(50, 2_000, n_looks).astype(int)
                cumulative = np.cumsum(rng.standard_normal(n_grid[-1]))
                estimates = cumulative[n_grid - 1] / n_grid
                variances = 1.0 / n_grid
                if sequential_p_values(estimates, variances, tau, alpha=ALPHA).stopped_early:
                    crossings += 1
            rate = crossings / n_sims
            se = np.sqrt(ALPHA * (1 - ALPHA) / n_sims)
            assert rate <= ALPHA + 4 * se, f"tau={tau} broke the guarantee: rate={rate:.4f}"

    def test_it_still_detects_a_real_effect(self):
        """A test that never rejects would satisfy the guarantee and be useless."""
        rng = np.random.default_rng(9)
        n_sims, n_looks = 400, 20
        true_effect, sigma, tau = 0.30, 1.0, 0.3
        detections = 0
        for _ in range(n_sims):
            n_grid = np.linspace(50, 3_000, n_looks).astype(int)
            cumulative = np.cumsum(rng.standard_normal(n_grid[-1]) + true_effect)
            estimates = cumulative[n_grid - 1] / n_grid
            variances = sigma**2 / n_grid
            if sequential_p_values(estimates, variances, tau, alpha=ALPHA).stopped_early:
                detections += 1
        power = detections / n_sims
        assert power > 0.90, f"always-valid power against a clear effect was only {power:.2f}"

    def test_early_stopping_happens_sooner_for_bigger_effects(self):
        rng = np.random.default_rng(10)
        n_grid = np.linspace(50, 3_000, 30).astype(int)

        def mean_stop(effect: float) -> float:
            stops = []
            for _ in range(200):
                cumulative = np.cumsum(rng.standard_normal(n_grid[-1]) + effect)
                estimates = cumulative[n_grid - 1] / n_grid
                seq = sequential_p_values(estimates, 1.0 / n_grid, tau=0.3, alpha=ALPHA)
                if seq.first_crossing is not None:
                    stops.append(seq.first_crossing)
            return float(np.mean(stops)) if stops else float("inf")

        assert mean_stop(0.60) < mean_stop(0.20)

    def test_p_value_is_conservative_relative_to_fixed_horizon(self):
        """Quantify the cost of monitoring, across a grid."""
        from scipy import stats

        for variance in (1e-2, 1e-4, 1e-6):
            for estimate in (0.01, 0.05, 0.2):
                se = np.sqrt(variance)
                fixed = float(2 * stats.norm.sf(abs(estimate / se)))
                sequential = always_valid_p_value(estimate, variance, tau=0.05)
                assert sequential >= fixed - 1e-12, (
                    f"always-valid p ({sequential:.3g}) below fixed-horizon "
                    f"({fixed:.3g}) at estimate={estimate}, var={variance}"
                )
