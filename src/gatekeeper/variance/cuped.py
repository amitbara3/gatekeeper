"""CUPED: variance reduction using a pre-experiment covariate.

The idea is simple and the payoff is large. If a pre-experiment covariate ``X``
correlates with the outcome ``Y``, then

    Y_adjusted = Y - theta * (X - mean(X)),    theta = cov(Y, X) / var(X)

has the same expectation as ``Y`` but its variance **multiplied** by ``1 - rho^2`` --
that is, a fraction ``rho^2`` of the variance removed. Run the ordinary test on
``Y_adjusted`` and every interval narrows for free.

Mind the direction, because the usual phrasing invites confusion: at rho = 0.7 the
variance is multiplied by 0.51, so the *reduction* is 0.49, not 0.51. That is worth
roughly a 2x larger sample.

**Why it stays unbiased.** ``X`` is measured *before* assignment, so it is independent
of treatment. Subtracting a function of ``X`` therefore shifts both arms' outcomes by
the same expected amount and leaves the difference untouched. That independence is
load-bearing, and it is precisely what fails for a post-treatment variable.

**The trap this module refuses to fall into.** On Cookie Cats the only numeric column
available is ``sum_gamerounds``, and using it as the covariate would appear to work
beautifully: it correlates strongly with retention, so the measured variance would drop
a lot. But it is measured *after* the player meets the gate, making it a mediator.
Adjusting for it changes the estimand and biases the result -- while the diagnostics
happily report a large "variance reduction". A number that looks like a win and is
actually bias is the most dangerous kind of output, so
:func:`estimate_cuped` raises :class:`PostTreatmentCovariateError` rather than
computing it (R1.7).

Consequently **CUPED cannot be applied to Cookie Cats at all** (PRD §6). It is
demonstrated here on synthetic data with a genuine pre-period covariate, and every
such estimate is tagged ``DataSource.SYNTHETIC``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from gatekeeper.data.schema import ExperimentData
from gatekeeper.frequentist.means import welch_test
from gatekeeper.types import (
    DataSource,
    EffectEstimate,
    Estimand,
    InsufficientData,
    Scale,
)

__all__ = ["CupedAdjustment", "cuped_adjust", "cuped_theta", "estimate_cuped"]


class CupedAdjustment(NamedTuple):
    """The adjustment and the diagnostics needed to judge whether it helped."""

    adjusted_control: np.ndarray
    adjusted_treatment: np.ndarray
    theta: float
    rho: float
    """Pooled correlation between covariate and outcome."""
    theoretical_reduction: float
    """``rho^2`` -- the *fraction of variance removed* that theory predicts.

    Note the direction carefully, because the usual phrasing invites an off-by-one-idea
    error. CUPED **multiplies** the variance by ``1 - rho^2``; the fraction it
    **removes** is therefore ``rho^2``. At rho = 0.7 the variance is multiplied by 0.51,
    i.e. 49% of it is eliminated, so the reduction is 0.49 and not 0.51.
    """
    achieved_reduction: float
    """Measured fraction of variance removed: ``1 - var_after / var_before``."""

    @property
    def effective_sample_size_multiplier(self) -> float:
        """How much larger an unadjusted experiment would need to be to match this.

        A 50% variance reduction is worth a 2x larger sample, which is the number
        that makes CUPED worth the trouble to a stakeholder.
        """
        if self.achieved_reduction >= 1.0:
            return math.inf
        return 1.0 / (1.0 - self.achieved_reduction)


def cuped_theta(outcome: np.ndarray, covariate: np.ndarray) -> float:
    """``theta = cov(Y, X) / var(X)``, estimated on the **pooled** sample.

    Pooling both arms is deliberate. Estimating ``theta`` separately per arm lets the
    treatment influence its own adjustment, which reintroduces exactly the dependence
    CUPED relies on avoiding. Since ``X`` is pre-treatment, pooling costs nothing and
    gives a more precise ``theta``.

    Raises
    ------
    InsufficientData
        If fewer than two observations, or the covariate has zero variance (in which
        case it explains nothing and cannot be divided by).
    """
    if outcome.size != covariate.size:
        raise ValueError(
            f"outcome and covariate must align elementwise, got {outcome.size} and {covariate.size}"
        )
    if outcome.size < 2:
        raise InsufficientData(f"need >= 2 observations, got {outcome.size}")

    var_x = float(covariate.var(ddof=1))
    if var_x == 0.0:
        raise InsufficientData(
            "covariate has zero variance, so it explains none of the outcome and "
            "theta is undefined; CUPED cannot help here"
        )
    cov_yx = float(np.cov(outcome, covariate, ddof=1)[0, 1])
    return cov_yx / var_x


def cuped_adjust(
    outcome_control: np.ndarray,
    covariate_control: np.ndarray,
    outcome_treatment: np.ndarray,
    covariate_treatment: np.ndarray,
) -> CupedAdjustment:
    """Apply the CUPED adjustment to both arms.

    ``theta`` and the covariate mean are computed on the pooled sample, then the same
    values are applied to both arms -- another consequence of ``X`` being
    pre-treatment, and what keeps the difference in means unbiased.

    Returns
    -------
    CupedAdjustment
        Adjusted outcomes plus theta, rho, and both the theoretical and achieved
        variance reduction. Comparing those two is the check that the implementation
        is doing what the theory says.
    """
    for name, out, cov in (
        ("control", outcome_control, covariate_control),
        ("treatment", outcome_treatment, covariate_treatment),
    ):
        if out.size != cov.size:
            raise ValueError(f"{name} arm: outcome and covariate lengths differ")
        if out.size < 2:
            raise InsufficientData(f"{name} arm needs >= 2 observations, got {out.size}")

    pooled_outcome = np.concatenate([outcome_control, outcome_treatment])
    pooled_covariate = np.concatenate([covariate_control, covariate_treatment])

    theta = cuped_theta(pooled_outcome, pooled_covariate)
    covariate_mean = float(pooled_covariate.mean())

    adjusted_c = outcome_control - theta * (covariate_control - covariate_mean)
    adjusted_t = outcome_treatment - theta * (covariate_treatment - covariate_mean)

    sd_y = float(pooled_outcome.std(ddof=1))
    sd_x = float(pooled_covariate.std(ddof=1))
    rho = (
        0.0
        if sd_y == 0.0 or sd_x == 0.0
        else float(np.corrcoef(pooled_outcome, pooled_covariate)[0, 1])
    )

    # Variance of the estimated difference, before and after.
    n_c, n_t = outcome_control.size, outcome_treatment.size
    var_before = outcome_control.var(ddof=1) / n_c + outcome_treatment.var(ddof=1) / n_t
    var_after = adjusted_c.var(ddof=1) / n_c + adjusted_t.var(ddof=1) / n_t
    achieved = 0.0 if var_before == 0 else float(1.0 - var_after / var_before)

    return CupedAdjustment(
        adjusted_control=adjusted_c,
        adjusted_treatment=adjusted_t,
        theta=theta,
        rho=rho,
        theoretical_reduction=rho**2,
        achieved_reduction=achieved,
    )


def estimate_cuped(
    data: ExperimentData,
    estimand: Estimand,
    covariate: str,
    *,
    alpha: float = 0.05,
    treatment_arm: str | None = None,
) -> EffectEstimate:
    """Estimate a treatment effect with CUPED variance reduction.

    Parameters
    ----------
    data
        The experiment.
    estimand
        What is being estimated. Absolute scale only -- see Raises.
    covariate
        Column to adjust on. **Must be pre-treatment.** Verified against the schema's
        ``post_treatment`` flags before anything is computed.
    alpha
        Two-sided significance level.
    treatment_arm
        Defaults to the single non-control arm.

    Returns
    -------
    EffectEstimate
        ``method="cuped_welch_t"``, with theta, rho, and both variance-reduction
        figures in ``diagnostics``.

    Raises
    ------
    PostTreatmentCovariateError
        If ``covariate`` is measured after assignment. This is the guard that stops
        ``sum_gamerounds`` from being used on Cookie Cats (R1.7).
    NotImplementedError
        On a relative-scale estimand: CUPED adjusts the outcome additively, so the
        relative interval would need care this module does not yet take.
    """
    # Order matters: refuse an invalid covariate before touching any data, so the
    # error is about the analysis plan rather than about the arithmetic.
    data.assert_pre_treatment(covariate)

    if estimand.scale is Scale.RELATIVE:
        raise NotImplementedError(
            "CUPED is implemented for the absolute scale only; the adjustment is "
            "additive on the outcome, and a relative interval on the adjusted values "
            "needs separate derivation"
        )

    control = data.control
    treatment = treatment_arm if treatment_arm is not None else data.treatment

    outcome_c = data.outcome(estimand.outcome, control)
    outcome_t = data.outcome(estimand.outcome, treatment)
    covariate_c = data.outcome(covariate, control)
    covariate_t = data.outcome(covariate, treatment)

    adjustment = cuped_adjust(outcome_c, covariate_c, outcome_t, covariate_t)
    result = welch_test(adjustment.adjusted_control, adjustment.adjusted_treatment, alpha=alpha)

    assumptions = [
        f"units ({data.schema.unit_col}) are independent",
        f"{covariate!r} is measured BEFORE assignment and is independent of treatment "
        "-- this is what keeps CUPED unbiased (R1.7)",
        "theta and the covariate mean are estimated on the POOLED sample, so treatment "
        "cannot influence its own adjustment",
        "the difference in MEANS is the quantity of interest",
        "central limit theorem applies at this sample size",
        "variances are NOT assumed equal (Welch on the adjusted outcomes)",
    ]
    if data.data_source is DataSource.REAL:
        assumptions.append("assignment was randomised, licensing a causal reading")
    else:
        assumptions.append(f"data is {data.data_source}, not a real experiment (R1.11)")

    # If theory and practice disagree, say so on the estimate rather than burying it.
    gap = abs(adjustment.achieved_reduction - adjustment.theoretical_reduction)
    if gap > 0.05:
        assumptions.append(
            f"WARNING: achieved variance reduction ({adjustment.achieved_reduction:.3f}) "
            f"differs from the theoretical 1 - rho^2 ({adjustment.theoretical_reduction:.3f}) "
            f"by {gap:.3f}; check the covariate's relationship with the outcome"
        )
    if adjustment.achieved_reduction < 0.01:
        assumptions.append(
            f"WARNING: CUPED did not reduce variance here (achieved "
            f"{adjustment.achieved_reduction:.4f}, rho={adjustment.rho:.4f}); the "
            "covariate carries essentially no information about the outcome, so the "
            "adjustment adds complexity for nothing"
        )

    return EffectEstimate(
        estimand=estimand,
        point=result.point,
        ci=result.ci,
        ci_level=1.0 - alpha,
        se=result.se,
        p_value=result.p_value,
        method="cuped_welch_t",
        assumptions=tuple(assumptions),
        data_source=data.data_source,
        n_per_arm={control: int(outcome_c.size), treatment: int(outcome_t.size)},
        diagnostics={
            "theta": adjustment.theta,
            "rho": adjustment.rho,
            "theoretical_reduction": adjustment.theoretical_reduction,
            "achieved_reduction": adjustment.achieved_reduction,
            "effective_n_multiplier": adjustment.effective_sample_size_multiplier,
            "mean_control_adjusted": float(adjustment.adjusted_control.mean()),
            "mean_treatment_adjusted": float(adjustment.adjusted_treatment.mean()),
            "df": result.df,
        },
    )
