"""Two-proportion tests for binary metrics.

**The pooled/unpooled split.** The p-value and the confidence interval use *different*
standard errors, and this is correct rather than sloppy:

- The **p-value** uses the **pooled** proportion. Under the null the two arms share a
  single rate, so the best estimate of it uses all the data. This is what
  ``statsmodels.proportions_ztest`` does, and our test cross-checks against it.
- The **interval** uses the **unpooled** SE. A CI describes plausible values of a
  *non-zero* difference, and assuming a common rate while estimating how much the
  rates differ would be incoherent.

A consequence worth knowing: with p very near alpha, the interval can just barely
include zero while the p-value sits just below alpha, or vice versa. That is not a
bug -- the two answer different questions with different variance estimates.

The p-value is identical on the absolute and relative scales, because the null
(``p_treatment == p_control``) is the same hypothesis either way. Only the interval
changes.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

from scipy import stats

from gatekeeper.data.schema import ExperimentData
from gatekeeper.types import (
    DataSource,
    EffectEstimate,
    Estimand,
    InsufficientData,
    Scale,
)

__all__ = ["ProportionResult", "estimate_two_proportion", "two_proportion_test"]

_MIN_EXPECTED_COUNT = 10
"""Below this many successes *or* failures in an arm, the normal approximation frays."""


class ProportionResult(NamedTuple):
    """Lightweight result of the core computation.

    Kept separate from :class:`~gatekeeper.types.EffectEstimate` so the calibration
    suite can run tens of thousands of simulations without paying for frame
    construction and validation on every draw.
    """

    point: float
    ci: tuple[float, float]
    se: float
    p_value: float
    z: float
    p_control: float
    p_treatment: float
    se_pooled: float


def two_proportion_test(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
    scale: Scale = Scale.ABSOLUTE,
    warn_small: bool = True,
) -> ProportionResult:
    """Compare two proportions. The core computation, on counts.

    Parameters
    ----------
    successes_control, n_control
        Control-arm successes and total.
    successes_treatment, n_treatment
        Treatment-arm successes and total.
    alpha
        Two-sided significance level.
    scale
        ``ABSOLUTE`` for the difference in rates, ``RELATIVE`` for
        ``(p_t - p_c) / p_c``.
    warn_small
        Emit a warning when an arm has fewer than 10 successes or failures. Disabled
        internally by the calibration suite, which runs many small simulations by
        design.

    Returns
    -------
    ProportionResult
        Point estimate, interval, SE, p-value, and intermediate quantities.

    Raises
    ------
    InsufficientData
        If an arm is empty, or -- on the relative scale -- the control rate is zero,
        which makes a ratio undefined.

    Notes
    -----
    The relative interval is built on the **log** scale (log relative risk, then
    exponentiated), not by a delta method on the ratio directly. A ratio is bounded
    below by zero and strongly right-skewed, so a symmetric interval on the raw ratio
    can extend below -100% relative change, which is impossible. The log scale cannot
    produce that.

    Assumptions
    -----------
    Independent Bernoulli units; normal approximation to the binomial; fixed sample
    size decided in advance.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_control < 1 or n_treatment < 1:
        raise InsufficientData(
            f"both arms need at least one unit, got {n_control} and {n_treatment}"
        )
    for label, s, n in (
        ("control", successes_control, n_control),
        ("treatment", successes_treatment, n_treatment),
    ):
        if not 0 <= s <= n:
            raise ValueError(f"{label} successes ({s}) must be in [0, n] with n={n}")

    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment

    if warn_small:
        thin = {
            label: (s, n - s)
            for label, s, n in (
                ("control", successes_control, n_control),
                ("treatment", successes_treatment, n_treatment),
            )
            if min(s, n - s) < _MIN_EXPECTED_COUNT
        }
        if thin:
            warnings.warn(
                f"normal approximation is unreliable: arm(s) {thin} have fewer than "
                f"{_MIN_EXPECTED_COUNT} successes or failures (shown as "
                "(successes, failures)). Prefer an exact test or the bootstrap.",
                UserWarning,
                stacklevel=2,
            )

    # p-value: pooled variance, because the null asserts a single shared rate.
    p_pool = (successes_control + successes_treatment) / (n_control + n_treatment)
    se_pooled = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_control + 1.0 / n_treatment))
    diff = p_t - p_c
    if se_pooled == 0.0:
        # Both arms all-success or all-failure: no evidence of any difference.
        z, p_value = 0.0, 1.0
    else:
        z = diff / se_pooled
        p_value = 2.0 * stats.norm.sf(abs(z))

    # Interval: unpooled variance, because it describes a non-zero difference.
    se_unpooled = math.sqrt(p_c * (1.0 - p_c) / n_control + p_t * (1.0 - p_t) / n_treatment)
    z_crit = stats.norm.isf(alpha / 2.0)

    if scale is Scale.ABSOLUTE:
        point = diff
        ci = (diff - z_crit * se_unpooled, diff + z_crit * se_unpooled)
        se_reported = se_unpooled
    else:
        if p_c == 0.0:
            raise InsufficientData(
                "relative lift is undefined when the control rate is zero; "
                "report the absolute difference instead"
            )
        if p_t == 0.0:
            # log(0) is undefined; the ratio is a hard zero, so lift is exactly -100%.
            point, ci, se_reported = -1.0, (-1.0, -1.0), 0.0
        else:
            log_rr = math.log(p_t / p_c)
            se_log = math.sqrt((1.0 - p_c) / (n_control * p_c) + (1.0 - p_t) / (n_treatment * p_t))
            point = math.exp(log_rr) - 1.0
            ci = (
                math.exp(log_rr - z_crit * se_log) - 1.0,
                math.exp(log_rr + z_crit * se_log) - 1.0,
            )
            se_reported = se_log

    return ProportionResult(
        point=point,
        ci=ci,
        se=se_reported,
        p_value=float(p_value),
        z=float(z),
        p_control=p_c,
        p_treatment=p_t,
        se_pooled=se_pooled,
    )


def estimate_two_proportion(
    data: ExperimentData,
    estimand: Estimand,
    *,
    alpha: float = 0.05,
    treatment_arm: str | None = None,
) -> EffectEstimate:
    """Estimate the effect on a binary metric from an :class:`ExperimentData`.

    Parameters
    ----------
    data
        The experiment. Must have passed the sanity gate (R1.3) -- this function does
        not re-check, since the gate is the caller's responsibility and an override
        must be recorded on the estimate.
    estimand
        What is being estimated, declared before estimation (R1.1).
    alpha
        Two-sided significance level; ``1 - alpha`` becomes the CI level.
    treatment_arm
        Which arm is the treatment. Defaults to the single non-control arm.

    Returns
    -------
    EffectEstimate
        With ``assumptions`` and ``data_source`` populated (R2.4, R1.11).
    """
    control = data.control
    treatment = treatment_arm if treatment_arm is not None else data.treatment
    metric = estimand.outcome

    values_c = data.outcome(metric, control)
    values_t = data.outcome(metric, treatment)
    for label, values in (("control", values_c), ("treatment", values_t)):
        unique = set(values.tolist())
        if not unique <= {0.0, 1.0}:
            raise ValueError(
                f"{metric!r} is not binary in the {label} arm (found values beyond "
                f"{{0, 1}}); use a means or bootstrap estimator for magnitude metrics"
            )

    result = two_proportion_test(
        int(values_c.sum()),
        int(values_c.size),
        int(values_t.sum()),
        int(values_t.size),
        alpha=alpha,
        scale=estimand.scale,
    )

    assumptions = [
        f"units ({data.schema.unit_col}) are independent",
        f"{metric!r} is a binary per-unit outcome",
        "normal approximation to the binomial is adequate at this sample size",
        "p-value uses the pooled variance (null: equal rates); interval uses unpooled",
        "sample size was fixed in advance; the data were not read repeatedly (R1.5)",
    ]
    if estimand.scale is Scale.RELATIVE:
        assumptions.append("relative interval built on the log-ratio scale")
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
        method="two_proportion_z",
        assumptions=tuple(assumptions),
        data_source=data.data_source,
        n_per_arm={control: int(values_c.size), treatment: int(values_t.size)},
        diagnostics={
            "p_control": result.p_control,
            "p_treatment": result.p_treatment,
            "z": result.z,
            "se_pooled": result.se_pooled,
            "se_unpooled": math.sqrt(
                result.p_control * (1 - result.p_control) / values_c.size
                + result.p_treatment * (1 - result.p_treatment) / values_t.size
            ),
        },
    )
