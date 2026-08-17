"""Power, sample size, MDE, and duration.

The pre-test half of the toolkit. Running a test that could never have detected the
effect you care about, then reading "not significant" as "no effect", is one of the
six failure modes this project exists to prevent (PRD §1).

**Design choice: one source of truth.** Only the ``power_*`` functions contain
statistical content. ``sample_size_*`` and ``mde_*`` are obtained by *numerically
inverting* them. Closed-form sample-size formulas exist, but they rest on their own
approximations, and keeping two derivations in sync is how the two quietly diverge.
Inversion costs microseconds and cannot disagree with the power function it inverts.
A test asserts round-tripping in both directions.

**Relationship between MDE and the practical threshold.** If the MDE exceeds the
smallest effect that would change a decision, a null result is uninformative -- it
cannot separate "no meaningful effect" from "underpowered". ``ExperimentSpec``
rejects that configuration outright.
"""

from __future__ import annotations

import math

from scipy import optimize, stats

from gatekeeper.types import InsufficientData

__all__ = [
    "duration_days",
    "mde_means",
    "mde_two_proportion",
    "power_means",
    "power_two_proportion",
    "sample_size_means",
    "sample_size_two_proportion",
]

_MAX_N = 10**9

_NCT_DF_LIMIT = 1_000.0
"""Above this many degrees of freedom, use the normal approximation instead of ``nct``.

``scipy.stats.nct`` returns NaN for large degrees of freedom -- at df ~ 12,000 with a
non-centrality of ~14 it fails outright, which broke the sample-size solver before this
branch existed. The t distribution converges to the normal anyway, so above the limit
the normal form is both more robust and accurate to well under 1e-4 in power. A test
asserts the two branches agree either side of the boundary.
"""


def _check_alpha_power(alpha: float, power: float | None = None) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if power is not None:
        if not 0.0 < power < 1.0:
            raise ValueError(f"power must be in (0, 1), got {power}")
        if power <= alpha:
            raise ValueError(
                f"target power ({power}) must exceed alpha ({alpha}); a test always has "
                "power at least alpha, since that is the rejection rate under the null"
            )


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------


def power_two_proportion(
    p_control: float,
    effect: float,
    n_control: int,
    n_treatment: int | None = None,
    *,
    alpha: float = 0.05,
) -> float:
    """Two-sided power of a two-proportion z-test.

    Parameters
    ----------
    p_control
        Control-arm proportion.
    effect
        True *absolute* difference, ``p_treatment - p_control``.
    n_control
        Units in the control arm.
    n_treatment
        Units in the treatment arm. Defaults to ``n_control``.
    alpha
        Two-sided significance level.

    Returns
    -------
    float
        Probability of rejecting the null when the true effect is ``effect``.

    Notes
    -----
    The rejection region uses the **pooled** variance, matching how the test is
    actually run under the null; the alternative distribution uses the **unpooled**
    variance, since under the alternative the two proportions genuinely differ. That
    asymmetry is not a subtlety to smooth over -- it is what makes power at
    ``effect=0`` come out to exactly ``alpha``, which is asserted in the tests.

    Both tails are included. The far tail is negligible for a large effect but not
    for a small one, and dropping it would put a floor of ``alpha/2`` on power.

    Assumptions
    -----------
    Normal approximation to the binomial; independent units; fixed n (no peeking).
    """
    _check_alpha_power(alpha)
    n_t = n_control if n_treatment is None else n_treatment
    if n_control < 1 or n_t < 1:
        raise InsufficientData(f"need >= 1 unit per arm, got {n_control} and {n_t}")
    if not 0.0 < p_control < 1.0:
        raise ValueError(f"p_control must be in (0, 1), got {p_control}")

    p_treatment = p_control + effect
    if not 0.0 <= p_treatment <= 1.0:
        raise ValueError(
            f"p_control + effect = {p_treatment:.4f} is outside [0, 1]; "
            f"an effect of {effect} is impossible from a base rate of {p_control}"
        )

    p_bar = (n_control * p_control + n_t * p_treatment) / (n_control + n_t)
    se_null = math.sqrt(p_bar * (1.0 - p_bar) * (1.0 / n_control + 1.0 / n_t))
    se_alt = math.sqrt(
        p_control * (1.0 - p_control) / n_control + p_treatment * (1.0 - p_treatment) / n_t
    )
    if se_alt == 0.0:
        # Degenerate: both arms deterministic. Any non-zero effect is detected.
        return 1.0 if effect != 0.0 else alpha

    crit = stats.norm.isf(alpha / 2.0) * se_null
    upper = stats.norm.sf((crit - effect) / se_alt)
    lower = stats.norm.cdf((-crit - effect) / se_alt)
    return float(upper + lower)


def sample_size_two_proportion(
    p_control: float,
    effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Units **per arm** needed to detect ``effect`` with the given power.

    Obtained by inverting :func:`power_two_proportion`, then rounding up and
    confirming the rounded value actually clears the target.
    """
    _check_alpha_power(alpha, power)
    if effect == 0.0:
        raise ValueError("cannot power a test for an effect of exactly zero")

    def deficit(n: float) -> float:
        return power_two_proportion(p_control, effect, round(n), alpha=alpha) - power

    if deficit(_MAX_N) < 0:
        raise InsufficientData(
            f"an effect of {effect} from a base rate of {p_control} cannot reach "
            f"power {power} at alpha {alpha} within {_MAX_N:,} units per arm"
        )

    lo = 2.0
    if deficit(lo) >= 0:
        return int(lo)
    # Bind to a typed local: scipy is untyped, so brentq returns Any and math.ceil
    # would propagate it into the declared int return.
    root: float = float(optimize.brentq(deficit, lo, float(_MAX_N), xtol=0.5))
    n = math.ceil(root)
    # brentq works on a step function here, so nudge up until the target is met.
    while n < _MAX_N and power_two_proportion(p_control, effect, n, alpha=alpha) < power:
        n += 1
    return n


def mde_two_proportion(
    p_control: float,
    n_control: int,
    n_treatment: int | None = None,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    direction: str = "increase",
) -> float:
    """Smallest absolute effect detectable at the given power -- the MDE.

    Parameters
    ----------
    direction
        ``"increase"`` for the smallest detectable positive effect,
        ``"decrease"`` for the negative one. They are **not** mirror images: the
        binomial variance depends on the proportion, so an increase and a decrease of
        the same magnitude are not equally detectable. Reporting a single symmetric
        MDE hides that.

    Returns
    -------
    float
        Signed MDE. Positive for ``"increase"``, negative for ``"decrease"``.
    """
    _check_alpha_power(alpha, power)
    if direction not in ("increase", "decrease"):
        raise ValueError(f"direction must be 'increase' or 'decrease', got {direction!r}")

    n_t = n_control if n_treatment is None else n_treatment
    sign = 1.0 if direction == "increase" else -1.0
    # Largest effect physically possible in this direction, held just inside [0, 1].
    limit = (1.0 - p_control) if sign > 0 else p_control
    hi = sign * limit * (1.0 - 1e-9)

    def deficit(effect: float) -> float:
        return power_two_proportion(p_control, effect, n_control, n_t, alpha=alpha) - power

    if deficit(hi) < 0:
        raise InsufficientData(
            f"with n={n_control}/{n_t} and a base rate of {p_control}, no achievable "
            f"{direction} reaches power {power} at alpha {alpha}. The experiment cannot "
            "answer this question at any effect size."
        )
    tiny = sign * 1e-12
    return float(optimize.brentq(deficit, tiny, hi, xtol=1e-10))


# ---------------------------------------------------------------------------
# Means
# ---------------------------------------------------------------------------


def power_means(
    effect: float,
    sd: float,
    n_control: int,
    n_treatment: int | None = None,
    *,
    alpha: float = 0.05,
) -> float:
    """Two-sided power of an equal-variance two-sample t-test.

    Uses the **non-central t** distribution rather than a normal approximation, so
    the result is exact for the equal-variance t-test and can be cross-checked
    against ``statsmodels.stats.power.TTestIndPower``.

    Parameters
    ----------
    effect
        True difference in means.
    sd
        Common within-arm standard deviation.
    n_control, n_treatment
        Units per arm; ``n_treatment`` defaults to ``n_control``.
    alpha
        Two-sided significance level.

    Notes
    -----
    Equal variance is assumed **here only**, because that is what makes the
    non-central t exact and gives a reference implementation to check against. It is
    not the project's analysis default: R1.12 requires Welch for the actual test. For
    planning purposes a single planning-time SD is the normal convention, since two
    reliable per-arm SDs are rarely available before the experiment runs.

    Assumptions
    -----------
    Normally distributed outcomes, equal variance across arms, independent units.
    """
    _check_alpha_power(alpha)
    n_t = n_control if n_treatment is None else n_treatment
    if n_control < 2 or n_t < 2:
        raise InsufficientData(f"need >= 2 units per arm, got {n_control} and {n_t}")
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")

    df = float(n_control + n_t - 2)
    # Non-centrality for a two-sample t with unequal group sizes. Equivalently
    # effect / se, where se = sd * sqrt(1/n_c + 1/n_t).
    nc = (effect / sd) * math.sqrt(n_control * n_t / (n_control + n_t))

    def _normal_branch() -> float:
        crit = stats.norm.isf(alpha / 2.0)
        return float(stats.norm.sf(crit - nc) + stats.norm.cdf(-crit - nc))

    if df > _NCT_DF_LIMIT:
        return _normal_branch()

    crit = stats.t.isf(alpha / 2.0, df)
    power = float(stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc))
    if not math.isfinite(power):
        # nct is fragile well inside its nominal domain; never propagate a NaN into a
        # root-finder or, worse, into a reported power figure.
        return _normal_branch()
    return power


def sample_size_means(
    effect: float,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Units **per arm** needed to detect a difference in means."""
    _check_alpha_power(alpha, power)
    if effect == 0.0:
        raise ValueError("cannot power a test for an effect of exactly zero")
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")

    def deficit(n: float) -> float:
        return power_means(effect, sd, round(n), alpha=alpha) - power

    if deficit(_MAX_N) < 0:
        raise InsufficientData(
            f"an effect of {effect} with sd {sd} cannot reach power {power} "
            f"within {_MAX_N:,} units per arm"
        )
    lo = 2.0
    if deficit(lo) >= 0:
        return int(lo)
    root: float = float(optimize.brentq(deficit, lo, float(_MAX_N), xtol=0.5))
    n = math.ceil(root)
    while n < _MAX_N and power_means(effect, sd, n, alpha=alpha) < power:
        n += 1
    return n


def mde_means(
    sd: float,
    n_control: int,
    n_treatment: int | None = None,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest detectable difference in means, as a positive magnitude.

    Unlike the proportion case this *is* symmetric: the normal variance does not
    depend on the mean, so an increase and a decrease of equal size are equally
    detectable.
    """
    _check_alpha_power(alpha, power)
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")
    n_t = n_control if n_treatment is None else n_treatment

    def deficit(effect: float) -> float:
        return power_means(effect, sd, n_control, n_t, alpha=alpha) - power

    hi = sd * 100.0
    if deficit(hi) < 0:
        raise InsufficientData(
            f"with n={n_control}/{n_t} and sd={sd}, power {power} is unreachable even "
            f"for an effect of {hi}"
        )
    return float(optimize.brentq(deficit, 1e-12, hi, xtol=1e-12))


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------


def duration_days(n_per_arm: int, n_arms: int, units_per_day: float) -> float:
    """Days needed to accrue ``n_per_arm`` units in each of ``n_arms`` arms.

    Returns a float; round **up** when reporting, and remember that experiments are
    usually run in whole weeks to avoid day-of-week composition effects.
    """
    if n_per_arm < 1:
        raise ValueError(f"n_per_arm must be >= 1, got {n_per_arm}")
    if n_arms < 2:
        raise ValueError(f"n_arms must be >= 2, got {n_arms}")
    if units_per_day <= 0:
        raise ValueError(f"units_per_day must be positive, got {units_per_day}")
    return (n_per_arm * n_arms) / units_per_day
