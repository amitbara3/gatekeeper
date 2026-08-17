"""Always-valid p-values via a mixture sequential probability ratio test (mSPRT).

**The problem.** A fixed-horizon p-value is only valid if you look **once**, at the
planned sample size. Check a dashboard every day and stop at the first ``p < 0.05`` and
the actual false-positive rate climbs far above 5% -- with ten looks it is roughly 20%.
This is the single most common way a well-run experiment produces a wrong answer, and
"just don't peek" has never worked, because people want to know.

**The fix.** An always-valid p-value satisfies

    P( there EXISTS n such that p_n <= alpha )  <=  alpha

so it can be monitored continuously and stopped the moment it drops below alpha,
without any correction for how often it was checked. The guarantee is over the whole
*path*, not a single point.

**How.** Mix the likelihood ratio over a prior on the effect size. For an estimate
``d`` with variance ``V`` and prior ``Normal(0, tau^2)``, the mixture likelihood ratio
against the null has a closed form::

    Lambda = sqrt( V / (V + tau^2) ) * exp( tau^2 * d^2 / (2 * V * (V + tau^2)) )

and ``p = min(1, 1/Lambda)``. Since ``Lambda`` is a martingale under the null, Ville's
inequality gives the guarantee above. Taking a running minimum keeps the sequence
monotone so an earlier low value is never forgotten.

**Why this rather than alpha-spending.** O'Brien-Fleming and Pocock boundaries need the
joint distribution of the sequential statistics, computed by recursive numerical
integration -- tractable, but a substantial piece of numerical work whose correctness is
hard to check independently. The mSPRT is closed-form, needs no pre-committed number of
looks, and its guarantee is directly verifiable by simulation, which is how this module
is validated. Alpha-spending is deferred, not forgotten.

**The cost.** Always-valid intervals are wider than fixed-horizon ones at the same n.
That is the honest price of being allowed to look whenever you like; it is not a defect.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from gatekeeper.types import InsufficientData

__all__ = [
    "SequentialResult",
    "always_valid_interval",
    "always_valid_p_value",
    "sequential_p_values",
    "suggest_tau",
]


class SequentialResult(NamedTuple):
    """Always-valid p-values along a monitoring path."""

    p_values: tuple[float, ...]
    """Running-minimum always-valid p-value at each look."""
    raw_p_values: tuple[float, ...]
    """Per-look values before the running minimum, for inspection."""
    tau: float
    first_crossing: int | None
    """Index of the first look at which ``p <= alpha``, or ``None``."""

    @property
    def stopped_early(self) -> bool:
        return self.first_crossing is not None


def suggest_tau(mde: float) -> float:
    """A default prior scale for the mSPRT mixture.

    ``tau`` expresses the effect size the test is *tuned* to detect: the mixture puts
    most of its mass on effects of that order, so the test is most powerful there.
    Using the spec's MDE is the principled choice -- it is already the answer to "what
    size of effect do we care about?", decided before the data arrived (R1.2).

    ``tau`` affects power, not validity. Any positive value preserves the always-valid
    guarantee; a badly-chosen one merely costs sensitivity. That robustness is worth
    knowing, since it means this parameter cannot invalidate a result.
    """
    if mde <= 0:
        raise ValueError(f"mde must be positive to derive tau, got {mde}")
    return float(mde)


def always_valid_p_value(estimate: float, variance: float, tau: float) -> float:
    """Always-valid p-value for one look.

    Parameters
    ----------
    estimate
        The estimated effect at this look (e.g. a difference in means).
    variance
        The **variance** of that estimate -- the square of its standard error, not the
        standard error itself.
    tau
        Prior scale for the mixture. See :func:`suggest_tau`.

    Returns
    -------
    float
        A p-value in (0, 1]. Monitored across looks it satisfies
        ``P(exists n: p_n <= alpha) <= alpha``.

    Raises
    ------
    InsufficientData
        If ``variance`` is not positive.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if variance <= 0:
        raise InsufficientData(
            f"variance must be positive to compute a likelihood ratio, got {variance}"
        )

    # Lambda = sqrt(V/(V+tau^2)) * exp( tau^2 d^2 / (2 V (V+tau^2)) )
    denom = variance + tau**2
    log_lambda = 0.5 * math.log(variance / denom) + (
        tau**2 * estimate**2 / (2.0 * variance * denom)
    )
    # Work in logs: at large effect sizes Lambda overflows a float outright.
    if log_lambda <= 0.0:
        return 1.0
    return float(min(1.0, math.exp(-log_lambda)))


def always_valid_interval(
    estimate: float, variance: float, tau: float, *, alpha: float = 0.05
) -> tuple[float, float]:
    """Always-valid confidence interval: the effects not rejected at this look.

    Obtained by inverting the test -- the set of null values ``d0`` for which
    ``always_valid_p_value(estimate - d0, ...) > alpha``. Because ``Lambda`` depends on
    the estimate only through ``(estimate - d0)^2``, the region is a symmetric interval
    with a closed-form half-width.

    Wider than a fixed-horizon interval at the same n, which is the price of
    unrestricted monitoring.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if variance <= 0:
        raise InsufficientData(f"variance must be positive, got {variance}")

    denom = variance + tau**2
    # Solve log_lambda = -log(alpha) for the squared deviation.
    target = -math.log(alpha) - 0.5 * math.log(variance / denom)
    if target <= 0:
        # Even a zero deviation would be rejected: no finite interval is excluded.
        return (-math.inf, math.inf)
    half_width = math.sqrt(target * 2.0 * variance * denom / tau**2)
    return (estimate - half_width, estimate + half_width)


def sequential_p_values(
    estimates: np.ndarray,
    variances: np.ndarray,
    tau: float,
    *,
    alpha: float = 0.05,
) -> SequentialResult:
    """Always-valid p-values along a monitoring path, with a running minimum.

    Parameters
    ----------
    estimates
        Effect estimate at each look, in order.
    variances
        Variance of the estimate at each look.
    tau
        Prior scale.
    alpha
        Level at which to record the first crossing.

    Returns
    -------
    SequentialResult

    Notes
    -----
    The running minimum matters. Without it, a p-value that dips below alpha and then
    rises again would let a decision be un-made by later data -- but the stopping
    decision was already available at the earlier look, so the guarantee has to account
    for it. Monotone p-values are what make "stop as soon as it crosses" coherent.
    """
    if estimates.size != variances.size:
        raise ValueError(
            f"estimates and variances must align, got {estimates.size} and {variances.size}"
        )
    if estimates.size == 0:
        raise InsufficientData("no looks supplied")

    raw = [
        always_valid_p_value(float(d), float(v), tau)
        for d, v in zip(estimates, variances, strict=True)
    ]
    running = list(np.minimum.accumulate(raw))

    first_crossing: int | None = None
    for i, p in enumerate(running):
        if p <= alpha:
            first_crossing = i
            break

    return SequentialResult(
        p_values=tuple(float(p) for p in running),
        raw_p_values=tuple(raw),
        tau=tau,
        first_crossing=first_crossing,
    )
