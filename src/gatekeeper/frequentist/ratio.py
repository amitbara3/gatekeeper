"""Delta-method inference for ratio metrics (F3.4).

Needed when the **unit of analysis differs from the unit of randomisation** (R1.13).
Randomise by user but measure per session, and sessions from the same user are
correlated -- treating them as independent shrinks the standard error and manufactures
significance. The fix is to keep the *user* as the unit and treat the metric as a
ratio of two per-user totals, then propagate variance through that ratio.

For a ratio ``R = mean(Y) / mean(X)`` over independent units, a first-order Taylor
expansion gives::

    Var(R) ~= (1 / (n * mean(X)^2)) * [ var(Y) - 2R*cov(Y,X) + R^2 * var(X) ]

The covariance term is the one that matters and the one most often dropped: Y and X
are strongly positively correlated (a user with more sessions has more clicks), and
ignoring ``cov(Y, X)`` **overstates** the variance substantially. Getting a
conservative answer by accident is still getting it wrong.

**Status on Cookie Cats: not applicable.** Every metric there is per-user and the
randomisation unit is the user, so R1.13 is already satisfied and a ratio estimator has
nothing to do. This module exists because the plan calls for it and because the
covariance subtlety is worth having implemented and tested; it is exercised on
synthetic data only.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy import stats

from gatekeeper.types import InsufficientData

__all__ = ["RatioDifference", "RatioEstimate", "ratio_difference", "ratio_variance"]


class RatioEstimate(NamedTuple):
    """A single arm's ratio and its delta-method variance."""

    ratio: float
    variance: float
    n: int
    mean_numerator: float
    mean_denominator: float

    @property
    def se(self) -> float:
        return math.sqrt(self.variance)


class RatioDifference(NamedTuple):
    """Difference between two arms' ratio metrics."""

    point: float
    ci: tuple[float, float]
    se: float
    p_value: float
    z: float
    control: RatioEstimate
    treatment: RatioEstimate


def ratio_variance(numerator: np.ndarray, denominator: np.ndarray) -> RatioEstimate:
    """Delta-method estimate of a ratio metric and its variance, for one arm.

    Parameters
    ----------
    numerator, denominator
        Per-unit totals, aligned elementwise. For "clicks per session" these are each
        user's click count and session count.

    Returns
    -------
    RatioEstimate

    Raises
    ------
    InsufficientData
        If fewer than two units, or the denominator mean is zero.
    ValueError
        If the arrays differ in length, or any denominator entry is negative.
    """
    if numerator.size != denominator.size:
        raise ValueError(
            f"numerator and denominator must align elementwise, got sizes "
            f"{numerator.size} and {denominator.size}"
        )
    n = numerator.size
    if n < 2:
        raise InsufficientData(f"ratio variance needs >= 2 units, got {n}")
    if np.any(denominator < 0):
        raise ValueError("denominator values must be non-negative")

    mean_y = float(numerator.mean())
    mean_x = float(denominator.mean())
    if mean_x == 0.0:
        raise InsufficientData(
            "denominator mean is zero, so the ratio is undefined; check that the "
            "denominator metric is populated for this arm"
        )

    ratio = mean_y / mean_x
    var_y = float(numerator.var(ddof=1))
    var_x = float(denominator.var(ddof=1))
    # np.cov returns the 2x2 matrix; [0, 1] is the covariance.
    cov_yx = float(np.cov(numerator, denominator, ddof=1)[0, 1])

    variance = (var_y - 2.0 * ratio * cov_yx + ratio**2 * var_x) / (n * mean_x**2)
    # Taylor truncation can push a near-zero variance slightly negative.
    variance = max(variance, 0.0)

    return RatioEstimate(
        ratio=ratio,
        variance=variance,
        n=n,
        mean_numerator=mean_y,
        mean_denominator=mean_x,
    )


def ratio_difference(
    numerator_control: np.ndarray,
    denominator_control: np.ndarray,
    numerator_treatment: np.ndarray,
    denominator_treatment: np.ndarray,
    *,
    alpha: float = 0.05,
) -> RatioDifference:
    """Difference between two arms' ratio metrics, treatment minus control.

    Arm variances add because the arms are independent by randomisation.

    Returns
    -------
    RatioDifference

    Assumptions
    -----------
    Independent units *within* the arrays (the point of the exercise -- one row per
    randomisation unit); independent arms; the delta-method linearisation is adequate,
    which needs the denominator mean to be comfortably away from zero.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    control = ratio_variance(numerator_control, denominator_control)
    treatment = ratio_variance(numerator_treatment, denominator_treatment)

    point = treatment.ratio - control.ratio
    se = math.sqrt(control.variance + treatment.variance)
    z_crit = stats.norm.isf(alpha / 2.0)

    if se == 0.0:
        return RatioDifference(
            point=point,
            ci=(point, point),
            se=0.0,
            p_value=1.0 if point == 0.0 else 0.0,
            z=0.0,
            control=control,
            treatment=treatment,
        )

    z = point / se
    return RatioDifference(
        point=point,
        ci=(point - z_crit * se, point + z_crit * se),
        se=se,
        p_value=float(2.0 * stats.norm.sf(abs(z))),
        z=float(z),
        control=control,
        treatment=treatment,
    )
