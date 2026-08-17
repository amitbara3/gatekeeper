"""Measure the peeking problem, and measure the fix.

Rules R1.5 forbids repeated looks without a sequential correction. This module exists so
that prohibition is backed by a number rather than an assertion: it simulates an
experiment monitored at several interim points and reports how often each approach
produces a false positive when there is genuinely no effect.

Expected outcome, written down before running it (R2.2):

- **naive** -- read a fixed-horizon p-value at every look and stop at the first
  ``p < alpha`` -- should reject far above alpha, rising with the number of looks.
- **always_valid** -- the mSPRT -- should stay at or below alpha regardless of how
  many looks are taken.
- **final_only** -- ignore the interim looks and read once at the end -- should sit at
  exactly alpha, confirming the simulation harness itself is unbiased.

Cookie Cats has no timestamps, so real accrual order is unobservable (PRD §6). Arrival
order here is *simulated*: units are generated in sequence and looks taken at
checkpoints. This measures the statistical phenomenon, not anything about the actual
Cookie Cats experiment, and every result is synthetic.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from gatekeeper.frequentist.proportions import two_proportion_test
from gatekeeper.sequential.always_valid import sequential_p_values

__all__ = ["PeekingResult", "PeekingStrategy", "simulate_peeking"]

PeekingStrategy = Literal["naive", "always_valid", "final_only"]


class PeekingResult(NamedTuple):
    """False-positive rate of a monitoring strategy under a true null."""

    strategy: PeekingStrategy
    rejection_rate: float
    alpha: float
    n_looks: int
    n_sims: int
    mean_stopping_look: float
    """Average look index at which a rejection occurred (1-based); NaN if none did."""

    @property
    def inflation_factor(self) -> float:
        """How many times the nominal error rate this strategy actually achieves."""
        return self.rejection_rate / self.alpha

    @property
    def controls_error_rate(self) -> bool:
        """Whether the rate is at or below nominal, within Monte Carlo noise."""
        se = float(np.sqrt(self.alpha * (1 - self.alpha) / self.n_sims))
        return bool(self.rejection_rate <= self.alpha + 3 * se)

    def describe(self) -> str:
        return (
            f"{self.strategy:13} rejection rate {self.rejection_rate:.4f} "
            f"({self.inflation_factor:.1f}x nominal alpha={self.alpha}) over "
            f"{self.n_looks} look(s), {self.n_sims:,} simulations"
        )


def simulate_peeking(
    strategy: PeekingStrategy,
    *,
    base_rate: float = 0.20,
    effect: float = 0.0,
    n_per_arm_final: int = 4_000,
    n_looks: int = 10,
    alpha: float = 0.05,
    n_sims: int = 2_000,
    tau: float | None = None,
    seed: int = 0,
) -> PeekingResult:
    """Simulate monitored experiments and measure the rejection rate.

    Parameters
    ----------
    strategy
        ``"naive"``, ``"always_valid"``, or ``"final_only"`` -- see module docstring.
    base_rate
        Control-arm conversion rate.
    effect
        True absolute effect. Leave at 0 to measure the false-positive rate.
    n_per_arm_final
        Units per arm at the final look.
    n_looks
        Number of equally spaced interim looks, the last being the final one.
    alpha
        Level at which a rejection is recorded.
    n_sims
        Number of simulated experiments.
    tau
        Prior scale for ``"always_valid"``. Defaults to the smallest effect this
        design could detect, which is the natural tuning point.
    seed
        RNG seed (R4.2).

    Returns
    -------
    PeekingResult

    Notes
    -----
    Accrual is simulated by drawing per-unit Bernoulli outcomes once and taking
    **cumulative** sums at each checkpoint, so the looks are properly nested -- look
    ``k+1`` contains every unit from look ``k``. Drawing fresh independent counts per
    look would make them independent and *understate* the peeking problem, since real
    peeking looks at overlapping data.
    """
    if n_looks < 1:
        raise ValueError(f"n_looks must be >= 1, got {n_looks}")
    if n_per_arm_final < n_looks:
        raise ValueError(
            f"n_per_arm_final ({n_per_arm_final}) must be at least n_looks ({n_looks})"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_arm_final // n_looks, n_per_arm_final, n_looks, dtype=int)
    p_treatment = base_rate + effect

    if tau is None:
        # Tune the mixture to the effect this design can just detect.
        from gatekeeper.design.power import mde_two_proportion

        tau = abs(mde_two_proportion(base_rate, n_per_arm_final, alpha=alpha))

    rejections = 0
    stopping_looks: list[int] = []

    for _ in range(n_sims):
        # Draw the whole stream once, then read nested prefixes.
        stream_c = rng.random(n_per_arm_final) < base_rate
        stream_t = rng.random(n_per_arm_final) < p_treatment
        cum_c = np.cumsum(stream_c)
        cum_t = np.cumsum(stream_t)

        if strategy == "final_only":
            n = int(checkpoints[-1])
            r = two_proportion_test(int(cum_c[n - 1]), n, int(cum_t[n - 1]), n, warn_small=False)
            if r.p_value < alpha:
                rejections += 1
                stopping_looks.append(n_looks)
            continue

        if strategy == "naive":
            for look_index, n in enumerate(checkpoints, start=1):
                n = int(n)
                r = two_proportion_test(
                    int(cum_c[n - 1]), n, int(cum_t[n - 1]), n, warn_small=False
                )
                if r.p_value < alpha:
                    rejections += 1
                    stopping_looks.append(look_index)
                    break
            continue

        if strategy != "always_valid":
            raise ValueError(f"unknown strategy {strategy!r}")

        estimates = np.empty(n_looks)
        variances = np.empty(n_looks)
        for i, n in enumerate(checkpoints):
            n = int(n)
            r = two_proportion_test(int(cum_c[n - 1]), n, int(cum_t[n - 1]), n, warn_small=False)
            estimates[i] = r.point
            # Guard the degenerate all-zero / all-one prefix, where se is 0.
            variances[i] = r.se**2 if r.se > 0 else np.nan

        usable = ~np.isnan(variances)
        if not usable.any():
            continue
        seq = sequential_p_values(estimates[usable], variances[usable], tau, alpha=alpha)
        if seq.stopped_early:
            rejections += 1
            assert seq.first_crossing is not None
            stopping_looks.append(seq.first_crossing + 1)

    return PeekingResult(
        strategy=strategy,
        rejection_rate=rejections / n_sims,
        alpha=alpha,
        n_looks=n_looks,
        n_sims=n_sims,
        mean_stopping_look=float(np.mean(stopping_looks)) if stopping_looks else float("nan"),
    )
