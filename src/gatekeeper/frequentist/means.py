"""Welch's t-test for magnitude metrics.

Welch is the **default**, not an option (R1.12). The equal-variance ("pooled")
t-test requires the two arms to share a variance, and a treatment that changes a
metric's level very often changes its spread too -- which is exactly the situation an
A/B test is built to create. Welch costs almost nothing when variances happen to be
equal and stays correct when they are not, so defaulting to pooled trades robustness
for nothing.

For heavily skewed metrics, note that Welch tests the difference in **means**, which
may not be the quantity worth testing. ``sum_gamerounds`` has a mean roughly four
times its median; a difference in means there is dominated by the tail. Prefer the
bootstrap or a rank-based test, and say in the spec which one was pre-registered.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy import stats

from gatekeeper.data.schema import ExperimentData
from gatekeeper.types import DataSource, EffectEstimate, Estimand, InsufficientData, Scale

__all__ = ["MeansResult", "estimate_welch", "welch_test"]


class MeansResult(NamedTuple):
    """Lightweight result of the core computation (see proportions.ProportionResult)."""

    point: float
    ci: tuple[float, float]
    se: float
    p_value: float
    t: float
    df: float
    mean_control: float
    mean_treatment: float


def welch_test(
    values_control: np.ndarray,
    values_treatment: np.ndarray,
    *,
    alpha: float = 0.05,
) -> MeansResult:
    """Welch's unequal-variance t-test. The core computation, on arrays.

    Parameters
    ----------
    values_control, values_treatment
        Per-unit metric values for each arm.
    alpha
        Two-sided significance level.

    Returns
    -------
    MeansResult
        Difference in means (treatment minus control), interval, SE, p-value, and the
        Welch-Satterthwaite degrees of freedom.

    Raises
    ------
    InsufficientData
        If either arm has fewer than two observations, since the sample variance is
        undefined below that.

    Notes
    -----
    Degrees of freedom use the Welch-Satterthwaite approximation::

        df = (v_c/n_c + v_t/n_t)^2 / [ (v_c/n_c)^2/(n_c-1) + (v_t/n_t)^2/(n_t-1) ]

    Sample variances use ``ddof=1``. Matches ``scipy.stats.ttest_ind(equal_var=False)``,
    which the reference tests assert.

    Assumptions
    -----------
    Independent units; means are the quantity of interest; the CLT has taken hold at
    this sample size (or the outcomes are normal). Notably does **not** assume equal
    variances.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n_c, n_t = values_control.size, values_treatment.size
    if n_c < 2 or n_t < 2:
        raise InsufficientData(
            f"Welch's t-test needs >= 2 observations per arm, got {n_c} and {n_t}; "
            "the sample variance is undefined otherwise"
        )

    mean_c = float(values_control.mean())
    mean_t = float(values_treatment.mean())
    var_c = float(values_control.var(ddof=1))
    var_t = float(values_treatment.var(ddof=1))

    diff = mean_t - mean_c
    se = math.sqrt(var_c / n_c + var_t / n_t)

    if se == 0.0:
        # Both arms constant. Either identical (no evidence) or a deterministic gap
        # about which a t-test has nothing to say -- report it honestly as such.
        return MeansResult(
            point=diff,
            ci=(diff, diff),
            se=0.0,
            p_value=1.0 if diff == 0.0 else 0.0,
            t=0.0 if diff == 0.0 else math.inf * (1 if diff > 0 else -1),
            df=float(n_c + n_t - 2),
            mean_control=mean_c,
            mean_treatment=mean_t,
        )

    term_c = var_c / n_c
    term_t = var_t / n_t
    df = (term_c + term_t) ** 2 / (term_c**2 / (n_c - 1) + term_t**2 / (n_t - 1))
    t_stat = diff / se
    p_value = 2.0 * stats.t.sf(abs(t_stat), df)
    t_crit = stats.t.isf(alpha / 2.0, df)

    return MeansResult(
        point=diff,
        ci=(diff - t_crit * se, diff + t_crit * se),
        se=se,
        p_value=float(p_value),
        t=float(t_stat),
        df=float(df),
        mean_control=mean_c,
        mean_treatment=mean_t,
    )


def estimate_welch(
    data: ExperimentData,
    estimand: Estimand,
    *,
    alpha: float = 0.05,
    treatment_arm: str | None = None,
) -> EffectEstimate:
    """Estimate a difference in means from an :class:`ExperimentData`.

    Relative scale is supported via the log-ratio of the two means, which requires
    both means to be positive.
    """
    control = data.control
    treatment = treatment_arm if treatment_arm is not None else data.treatment
    metric = estimand.outcome

    values_c = data.outcome(metric, control)
    values_t = data.outcome(metric, treatment)
    result = welch_test(values_c, values_t, alpha=alpha)

    point, ci, se_reported = result.point, result.ci, result.se
    if estimand.scale is Scale.RELATIVE:
        if result.mean_control <= 0 or result.mean_treatment <= 0:
            raise InsufficientData(
                f"relative scale needs both arm means positive, got "
                f"control={result.mean_control:.6g}, treatment={result.mean_treatment:.6g}; "
                "report the absolute difference instead"
            )
        # Delta-method SE of log(mean_t) - log(mean_c), then exponentiate.
        se_log = math.sqrt(
            values_c.var(ddof=1) / (values_c.size * result.mean_control**2)
            + values_t.var(ddof=1) / (values_t.size * result.mean_treatment**2)
        )
        log_ratio = math.log(result.mean_treatment / result.mean_control)
        z_crit = stats.t.isf(alpha / 2.0, result.df)
        point = math.exp(log_ratio) - 1.0
        ci = (
            math.exp(log_ratio - z_crit * se_log) - 1.0,
            math.exp(log_ratio + z_crit * se_log) - 1.0,
        )
        se_reported = se_log

    assumptions = [
        f"units ({data.schema.unit_col}) are independent",
        "the difference in MEANS is the quantity of interest",
        "central limit theorem applies at this sample size",
        "variances are NOT assumed equal (Welch-Satterthwaite df)",
        "sample size was fixed in advance (R1.5)",
    ]
    if estimand.scale is Scale.RELATIVE:
        assumptions.append("relative interval via the delta method on the log ratio")
    if data.data_source is DataSource.REAL:
        assumptions.append("assignment was randomised, licensing a causal reading")
    else:
        assumptions.append(f"data is {data.data_source}, not a real experiment (R1.11)")

    # A heavily skewed metric makes the mean a poor summary; say so on the estimate
    # itself rather than leaving it for the reader to notice.
    median_c = float(np.median(values_c))
    if median_c > 0 and result.mean_control / median_c > 2.0:
        assumptions.append(
            f"WARNING: {metric!r} is heavily skewed (control mean/median = "
            f"{result.mean_control / median_c:.2f}); the mean is tail-driven and a "
            "bootstrap or rank-based test is likely more informative (R1.12)"
        )

    return EffectEstimate(
        estimand=estimand,
        point=point,
        ci=ci,
        ci_level=1.0 - alpha,
        se=se_reported,
        p_value=result.p_value,
        method="welch_t",
        assumptions=tuple(assumptions),
        data_source=data.data_source,
        n_per_arm={control: int(values_c.size), treatment: int(values_t.size)},
        diagnostics={
            "mean_control": result.mean_control,
            "mean_treatment": result.mean_treatment,
            "t": result.t,
            "df": result.df,
            "se_difference": result.se,
        },
    )
