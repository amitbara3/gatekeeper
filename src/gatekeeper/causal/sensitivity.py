"""Sensitivity analysis: how strong would unmeasured confounding have to be?

Every observational estimate rests on an assumption nothing in the data can check --
that all confounders were measured. Sensitivity analysis does not rescue that assumption.
It quantifies what it would take to overturn the conclusion, which converts an
unfalsifiable claim into a falsifiable one: instead of "we assume no unmeasured
confounding", you get "a confounder would need to be associated with both treatment and
outcome by a risk ratio of at least 3.2, and we think that is implausible because ...".

The **E-value** is the minimum strength of association -- on the risk-ratio scale, with
both treatment and outcome -- that an unmeasured confounder would need in order to explain
away the observed effect. Closed form (VanderWeele & Ding 2017)::

    E = RR + sqrt( RR * (RR - 1) )

An E-value near 1 means the finding is fragile: a weak confounder suffices. A large E-value
means only an implausibly strong one would do.

**What an E-value is not.** It is not a probability that confounding exists, and not
evidence that it does not. It is a statement about magnitude, and reporting it alongside a
list of plausible confounders and their likely strengths is what makes it useful. Reporting
the number alone is how it gets misused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["EValue", "e_value", "e_value_from_difference"]


@dataclass(frozen=True, slots=True)
class EValue:
    """An E-value for a point estimate and, optionally, for the interval bound."""

    point: float
    """E-value for the point estimate."""
    bound: float | None
    """E-value for the confidence limit nearer the null. ``None`` if not supplied.

    Usually the more useful of the two: it asks what would be needed to push the interval
    across the null rather than just the point estimate, which is the weaker claim.
    """
    risk_ratio: float
    """The risk ratio the E-value was computed from."""

    @property
    def is_fragile(self) -> bool:
        """Whether a modest confounder (RR < 1.5 with both) could explain the result."""
        reference = self.bound if self.bound is not None else self.point
        return reference < 1.5

    def describe(self) -> str:
        bound = "n/a" if self.bound is None else f"{self.bound:.2f}"
        verdict = "FRAGILE" if self.is_fragile else "robust to modest confounding"
        return (
            f"E-value {self.point:.2f} (interval bound {bound}) from RR "
            f"{self.risk_ratio:.4f} -- {verdict}. An unmeasured confounder would need "
            f"associations at least this strong with BOTH treatment and outcome to "
            "explain the finding away."
        )


def e_value(risk_ratio: float, ci_bound: float | None = None) -> EValue:
    """E-value for a risk ratio.

    Parameters
    ----------
    risk_ratio
        The observed risk ratio. Must be positive. Ratios below 1 are inverted first, since
        the E-value is symmetric in the direction of effect -- explaining away a protective
        effect takes the same strength as explaining away a harmful one of equal magnitude.
    ci_bound
        The confidence limit **closer to the null**. Supplying it gives the more demanding
        and more useful E-value.

    Returns
    -------
    EValue

    Examples
    --------
    A risk ratio of 2 needs a confounder at RR ~ 3.41 with both treatment and outcome:

    >>> round(e_value(2.0).point, 4)
    3.4142

    A null result has an E-value of exactly 1 -- no confounding at all is needed to
    explain nothing:

    >>> e_value(1.0).point
    1.0
    """
    if risk_ratio <= 0:
        raise ValueError(f"risk ratio must be positive, got {risk_ratio}")

    def compute(rr: float) -> float:
        # Invert protective effects; the scale is symmetric.
        if rr < 1.0:
            rr = 1.0 / rr
        return rr + math.sqrt(rr * (rr - 1.0))

    bound_value: float | None = None
    if ci_bound is not None:
        if ci_bound <= 0:
            raise ValueError(f"ci_bound must be positive, got {ci_bound}")
        # If the interval crosses the null, no confounding is needed at all.
        crosses = (risk_ratio > 1.0 and ci_bound <= 1.0) or (risk_ratio < 1.0 and ci_bound >= 1.0)
        bound_value = 1.0 if crosses else compute(ci_bound)

    return EValue(point=compute(risk_ratio), bound=bound_value, risk_ratio=risk_ratio)


def e_value_from_difference(
    difference: float,
    baseline_rate: float,
    ci_bound: float | None = None,
) -> EValue:
    """E-value for a **risk difference** on a binary outcome.

    Converts the difference to a risk ratio against ``baseline_rate`` first, since the
    E-value is defined on the ratio scale. A 0.8pp drop from a 19% base rate is a risk
    ratio of 0.958, which is a much less impressive-sounding number than "0.8 percentage
    points" -- and the E-value that follows will be correspondingly modest, which is the
    honest reading.

    Parameters
    ----------
    difference
        Absolute difference in rates (treatment minus control).
    baseline_rate
        Control-arm rate. Must be in (0, 1).
    ci_bound
        Confidence limit on the difference, closer to the null.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
    treated_rate = baseline_rate + difference
    if not 0.0 < treated_rate < 1.0:
        raise ValueError(
            f"baseline + difference = {treated_rate:.4f} is outside (0, 1); the "
            "difference is impossible from this baseline"
        )

    rr = treated_rate / baseline_rate
    bound_rr = None
    if ci_bound is not None:
        bound_treated = baseline_rate + ci_bound
        if not 0.0 < bound_treated < 1.0:
            raise ValueError(f"baseline + ci_bound = {bound_treated:.4f} is outside (0, 1)")
        bound_rr = bound_treated / baseline_rate

    return e_value(rr, bound_rr)
