"""Bootstrap intervals for a difference in means: percentile and BCa.

The right tool for ``sum_gamerounds``, whose distribution is severely right-skewed.
Welch's t-test still works there by the CLT at large n, but the bootstrap makes no
distributional claim at all, which is a cleaner story when a metric's mean is
tail-driven.

**Percentile vs BCa.** The percentile interval is simply the middle ``1-alpha`` of the
bootstrap distribution. It is easy to explain but biased when the statistic's
distribution is skewed. BCa (bias-corrected and accelerated) applies two corrections:
``z0`` for median bias -- where the bootstrap distribution sits relative to the
observed estimate -- and ``a`` (acceleration) for how the variance changes with the
parameter. On skewed data BCa's coverage is markedly closer to nominal, which is why
it is the default here, and the calibration suite is what actually demonstrates that
rather than taking it on trust.

**Scope.** BCa is implemented specifically for the **difference in means**, whose
jackknife influence values have a closed form (see :func:`_influence_values`), making
it exact and O(n) instead of O(n²). Since a proportion is the mean of an indicator,
this covers binary metrics too. A general-statistic percentile bootstrap is also
provided.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
from scipy import stats

from gatekeeper.data.schema import ExperimentData
from gatekeeper.types import DataSource, EffectEstimate, Estimand, InsufficientData, Scale

__all__ = [
    "BootstrapResult",
    "bootstrap_mean_difference",
    "bootstrap_statistic",
    "estimate_bootstrap",
]

BootstrapMethod = Literal["bca", "percentile"]

_MAX_RESAMPLE_CELLS = 8_000_000
"""Cap on ``chunk * n`` index draws held at once, to bound peak memory."""


class BootstrapResult(NamedTuple):
    """Result of a bootstrap, with the BCa corrections exposed for inspection."""

    point: float
    ci: tuple[float, float]
    se: float
    p_value: float
    method: BootstrapMethod
    n_resamples: int
    z0: float
    acceleration: float

    @property
    def is_skewed(self) -> bool:
        """Whether the bias correction is large enough to matter.

        ``|z0| > 0.1`` means the bootstrap distribution's median sits noticeably off
        the observed estimate -- the situation in which a percentile interval starts
        to mislead and BCa earns its extra machinery.
        """
        return abs(self.z0) > 0.1


def _resample_means(values: np.ndarray, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the mean, resampling in memory-bounded chunks.

    A naive ``(n_resamples, n)`` index matrix is 4.5e8 entries for 45k units and
    10k resamples, which is several gigabytes. Chunking keeps peak memory flat
    regardless of n while producing identical draws for a given generator state.
    """
    n = values.size
    chunk = max(1, min(n_resamples, _MAX_RESAMPLE_CELLS // max(n, 1)))
    out = np.empty(n_resamples, dtype=float)
    done = 0
    while done < n_resamples:
        size = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(size, n))
        out[done : done + size] = values[idx].mean(axis=1)
        done += size
    return out


def _influence_values(values_control: np.ndarray, values_treatment: np.ndarray) -> np.ndarray:
    """Jackknife influence values for the difference in means.

    For ``theta = mean_t - mean_c``, the leave-one-out algebra collapses to a closed
    form. Writing ``U_i = (n_k - 1)(theta_hat - theta_hat_without_i)``:

    - control observation:   ``U_i = mean_c - x_i``
    - treatment observation: ``U_j = x_j - mean_t``

    So the influence values are just the centred observations, sign-flipped in the
    control arm. Deriving this is what turns BCa from an O(n^2) numerical jackknife
    into an O(n) vectorised expression -- and a test checks it against the brute-force
    leave-one-out computation on small inputs.
    """
    return np.concatenate(
        [
            values_control.mean() - values_control,
            values_treatment - values_treatment.mean(),
        ]
    )


def bootstrap_mean_difference(
    values_control: np.ndarray,
    values_treatment: np.ndarray,
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    method: BootstrapMethod = "bca",
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Bootstrap the difference in means (treatment minus control).

    Parameters
    ----------
    values_control, values_treatment
        Per-unit metric values.
    alpha
        Two-sided level; the interval covers ``1 - alpha``.
    n_resamples
        Bootstrap replicates. 10,000 is enough for a 95% interval; a 99% interval
        wants more, since the tails are estimated from fewer replicates.
    method
        ``"bca"`` (default) or ``"percentile"``.
    seed, rng
        Exactly one may be given, or neither. Randomness is always explicit and the
        seed is recorded (R4.2).

    Returns
    -------
    BootstrapResult

    Raises
    ------
    InsufficientData
        If either arm has fewer than two observations.
    ValueError
        If both ``seed`` and ``rng`` are supplied.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be >= 100 for a meaningful interval, got {n_resamples}")
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")
    n_c, n_t = values_control.size, values_treatment.size
    if n_c < 2 or n_t < 2:
        raise InsufficientData(f"bootstrap needs >= 2 observations per arm, got {n_c} and {n_t}")

    generator = rng if rng is not None else np.random.default_rng(seed)

    observed = float(values_treatment.mean() - values_control.mean())
    boot = _resample_means(values_treatment, n_resamples, generator) - _resample_means(
        values_control, n_resamples, generator
    )
    se = float(boot.std(ddof=1))

    # Two-sided achieved significance level: how far into the tail zero sits.
    frac_le = float(np.mean(boot <= 0.0))
    frac_ge = float(np.mean(boot >= 0.0))
    p_value = min(1.0, 2.0 * min(frac_le, frac_ge))
    p_value = max(p_value, 1.0 / n_resamples)  # cannot resolve below 1/B

    if se == 0.0:
        return BootstrapResult(
            point=observed,
            ci=(observed, observed),
            se=0.0,
            p_value=1.0 if observed == 0.0 else 1.0 / n_resamples,
            method=method,
            n_resamples=n_resamples,
            z0=0.0,
            acceleration=0.0,
        )

    z_lo = stats.norm.ppf(alpha / 2.0)
    z_hi = stats.norm.ppf(1.0 - alpha / 2.0)

    if method == "percentile":
        ci = (
            float(np.percentile(boot, 100.0 * alpha / 2.0)),
            float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0))),
        )
        return BootstrapResult(
            point=observed,
            ci=ci,
            se=se,
            p_value=p_value,
            method="percentile",
            n_resamples=n_resamples,
            z0=0.0,
            acceleration=0.0,
        )

    # --- BCa -----------------------------------------------------------------
    # Bias correction: where the observed estimate falls in the bootstrap distribution.
    prop_below = float(np.mean(boot < observed))
    # Clip so an extreme proportion does not send z0 to infinity.
    prop_below = min(max(prop_below, 1.0 / (2 * n_resamples)), 1.0 - 1.0 / (2 * n_resamples))
    z0 = float(stats.norm.ppf(prop_below))

    influence = _influence_values(values_control, values_treatment)
    sum_sq = float(np.sum(influence**2))
    acceleration = 0.0 if sum_sq == 0.0 else float(np.sum(influence**3) / (6.0 * sum_sq**1.5))

    def adjusted(z: float) -> float:
        denom = 1.0 - acceleration * (z0 + z)
        if denom == 0.0:
            return float(stats.norm.cdf(z0 + z))
        return float(stats.norm.cdf(z0 + (z0 + z) / denom))

    a_lo, a_hi = adjusted(z_lo), adjusted(z_hi)
    if not (0.0 < a_lo < a_hi < 1.0):
        # Corrections degenerated (tiny n, or an extreme distribution). Fall back to
        # the percentile interval rather than emit a nonsensical one -- and say so.
        ci = (
            float(np.percentile(boot, 100.0 * alpha / 2.0)),
            float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0))),
        )
        return BootstrapResult(
            point=observed,
            ci=ci,
            se=se,
            p_value=p_value,
            method="percentile",
            n_resamples=n_resamples,
            z0=z0,
            acceleration=acceleration,
        )

    ci = (
        float(np.percentile(boot, 100.0 * a_lo)),
        float(np.percentile(boot, 100.0 * a_hi)),
    )
    return BootstrapResult(
        point=observed,
        ci=ci,
        se=se,
        p_value=p_value,
        method="bca",
        n_resamples=n_resamples,
        z0=z0,
        acceleration=acceleration,
    )


def bootstrap_statistic(
    values_control: np.ndarray,
    values_treatment: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Percentile bootstrap of ``statistic(treatment) - statistic(control)``.

    For any statistic other than the mean -- a median, a trimmed mean, a quantile.
    Percentile only: BCa's acceleration needs influence values, which are
    statistic-specific and have no closed form in general.

    Parameters
    ----------
    statistic
        Maps a 1-D array to a scalar, e.g. ``np.median``.
    """
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be >= 100, got {n_resamples}")
    n_c, n_t = values_control.size, values_treatment.size
    if n_c < 2 or n_t < 2:
        raise InsufficientData(f"need >= 2 observations per arm, got {n_c} and {n_t}")

    generator = rng if rng is not None else np.random.default_rng(seed)
    observed = float(statistic(values_treatment)) - float(statistic(values_control))

    boot = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        rc = values_control[generator.integers(0, n_c, size=n_c)]
        rt = values_treatment[generator.integers(0, n_t, size=n_t)]
        boot[b] = float(statistic(rt)) - float(statistic(rc))

    frac_le = float(np.mean(boot <= 0.0))
    frac_ge = float(np.mean(boot >= 0.0))
    p_value = max(min(1.0, 2.0 * min(frac_le, frac_ge)), 1.0 / n_resamples)

    return BootstrapResult(
        point=observed,
        ci=(
            float(np.percentile(boot, 100.0 * alpha / 2.0)),
            float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0))),
        ),
        se=float(boot.std(ddof=1)),
        p_value=p_value,
        method="percentile",
        n_resamples=n_resamples,
        z0=0.0,
        acceleration=0.0,
    )


def estimate_bootstrap(
    data: ExperimentData,
    estimand: Estimand,
    *,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    method: BootstrapMethod = "bca",
    seed: int = 0,
    treatment_arm: str | None = None,
) -> EffectEstimate:
    """Bootstrap the effect on a magnitude metric from an :class:`ExperimentData`."""
    if estimand.scale is Scale.RELATIVE:
        raise NotImplementedError(
            "relative-scale bootstrap is not implemented; bootstrap the absolute "
            "difference, or use estimate_welch for a relative difference in means"
        )

    control = data.control
    treatment = treatment_arm if treatment_arm is not None else data.treatment
    values_c = data.outcome(estimand.outcome, control)
    values_t = data.outcome(estimand.outcome, treatment)

    result = bootstrap_mean_difference(
        values_c,
        values_t,
        alpha=alpha,
        n_resamples=n_resamples,
        method=method,
        seed=seed,
    )

    assumptions = [
        f"units ({data.schema.unit_col}) are independent and identically distributed within arm",
        "the difference in MEANS is the quantity of interest",
        "the empirical distribution is a good stand-in for the population "
        "(no distributional form assumed)",
        f"{result.n_resamples:,} resamples; interval method: {result.method}",
        "sample size was fixed in advance (R1.5)",
    ]
    if result.method == "bca":
        assumptions.append(
            f"BCa corrections: z0={result.z0:.4f} (median bias), "
            f"a={result.acceleration:.4f} (acceleration)"
        )
    if method == "bca" and result.method == "percentile":
        assumptions.append(
            "WARNING: BCa was requested but its corrections degenerated; fell back "
            "to the percentile interval"
        )
    if data.data_source is DataSource.REAL:
        assumptions.append("assignment was randomised, licensing a causal reading")
    else:
        assumptions.append(f"data is {data.data_source}, not a real experiment (R1.11)")

    return EffectEstimate(
        estimand=estimand,
        point=result.point,
        ci=result.ci,
        ci_level=1.0 - alpha,
        se=result.se,
        p_value=result.p_value,
        method=f"bootstrap_{result.method}",
        assumptions=tuple(assumptions),
        data_source=data.data_source,
        n_per_arm={control: int(values_c.size), treatment: int(values_t.size)},
        diagnostics={
            "mean_control": float(values_c.mean()),
            "mean_treatment": float(values_t.mean()),
            "bootstrap_se": result.se,
            "z0": result.z0,
            "acceleration": result.acceleration,
            "n_resamples": float(result.n_resamples),
        },
        seed=seed,
    )


def _brute_force_influence(values_control: np.ndarray, values_treatment: np.ndarray) -> np.ndarray:
    """Numerical leave-one-out influence values. Test oracle for the closed form.

    Deliberately the slow, obvious implementation: it exists only so a test can
    confirm that the vectorised algebra in :func:`_influence_values` is right.
    """
    theta = float(values_treatment.mean() - values_control.mean())
    out: list[float] = []
    n_c, n_t = values_control.size, values_treatment.size
    for i in range(n_c):
        without = np.delete(values_control, i)
        out.append((n_c - 1) * (theta - (values_treatment.mean() - without.mean())))
    for j in range(n_t):
        without = np.delete(values_treatment, j)
        out.append((n_t - 1) * (theta - (without.mean() - values_control.mean())))
    return np.array(out, dtype=float)
