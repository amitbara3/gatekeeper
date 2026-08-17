"""Propensity scores: estimation, IPW, stabilised weights, and overlap diagnostics.

The propensity score is ``e(x) = P(treated | x)``. Weighting each unit by the inverse of
its probability of receiving the treatment it actually got reconstructs a pseudo-population
in which treatment is independent of the covariates -- which is what randomisation would
have given us for free.

**The assumptions are the whole story**, and they are not testable from the data:

- *Conditional ignorability*: given ``x``, treatment assignment is independent of the
  potential outcomes. In plain terms, ``x`` contains every confounder. Nothing in the data
  can verify this; the ``unobserved`` regime exists to show what happens when it fails.
- *Positivity / overlap*: every unit has a propensity strictly between 0 and 1. Where it
  is nearly violated, weights explode and the estimate is driven by a handful of units.

**Why overlap diagnostics are not optional.** A propensity near 0 or 1 produces a weight
in the hundreds, and a single such unit can dominate the estimate while every summary
statistic still looks reasonable. So ``estimate_ipw`` reports the maximum weight and the
effective sample size, warns when overlap is poor, and **raises** when it is absent.
Quietly trimming to make the number look better is the one response that is not available.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

from gatekeeper.data.schema import ExperimentData
from gatekeeper.types import (
    AssumptionViolation,
    DataSource,
    EffectEstimate,
    Estimand,
    EstimandTarget,
    InsufficientData,
)

__all__ = [
    "PropensityFit",
    "estimate_ipw",
    "estimate_naive_difference",
    "fit_propensity",
]

_MIN_OVERLAP = 0.01
"""Below this propensity, weights exceed 100 and the estimate stops being credible."""


@dataclass(frozen=True, slots=True)
class PropensityFit:
    """Fitted propensity scores plus the diagnostics needed to trust them."""

    scores: np.ndarray
    treated: np.ndarray
    covariates: tuple[str, ...]
    weights: np.ndarray
    """Stabilised inverse-probability weights."""

    @property
    def min_score(self) -> float:
        return float(self.scores.min())

    @property
    def max_score(self) -> float:
        return float(self.scores.max())

    @property
    def max_weight(self) -> float:
        return float(self.weights.max())

    @property
    def effective_sample_size(self) -> float:
        """Kish's ESS: ``(sum w)^2 / sum w^2``.

        The single most useful overlap diagnostic. If 20,000 units give an ESS of 300,
        the estimate rests on 300 units' worth of information however large the frame
        looks.
        """
        total = float(self.weights.sum())
        sum_sq = float(np.sum(self.weights**2))
        return 0.0 if sum_sq == 0 else total**2 / sum_sq

    @property
    def ess_fraction(self) -> float:
        return self.effective_sample_size / self.scores.size

    @property
    def has_good_overlap(self) -> bool:
        return self.min_score >= 0.05 and self.max_score <= 0.95

    def describe(self) -> str:
        return (
            f"propensity in [{self.min_score:.4f}, {self.max_score:.4f}]  "
            f"max weight {self.max_weight:.1f}  "
            f"ESS {self.effective_sample_size:,.0f}/{self.scores.size:,} "
            f"({self.ess_fraction:.1%})"
        )


def fit_propensity(
    data: ExperimentData,
    covariates: Sequence[str],
    *,
    treatment_column: str | None = None,
    stabilised: bool = True,
) -> PropensityFit:
    """Fit ``P(treated | covariates)`` by logistic regression.

    Parameters
    ----------
    data
        The experiment.
    covariates
        Pre-treatment covariates to condition on. **Each is checked against the schema's
        post-treatment flags** -- conditioning on a mediator is R1.7's error whether it
        happens in CUPED or here.
    treatment_column
        Column holding the received-treatment indicator. Defaults to the variant column,
        binarised against the control label.
    stabilised
        Use stabilised weights (multiply by the marginal treatment probability). Stabilised
        weights have the same expectation but far lower variance when propensities approach
        0 or 1, which is exactly when it matters.

    Raises
    ------
    PostTreatmentCovariateError
        If any covariate is measured after assignment.
    AssumptionViolation
        If positivity fails outright -- some unit has essentially no chance of the arm it
        received, so no reweighting can rescue it.
    """
    if not covariates:
        raise ValueError("propensity model needs at least one covariate")
    for name in covariates:
        data.assert_pre_treatment(name)

    frame = data.frame
    if treatment_column is None:
        treated = (frame[data.schema.variant_col] != data.control).to_numpy(dtype=float)
    else:
        treated = frame[treatment_column].to_numpy(dtype=float)
    if not set(np.unique(treated)) <= {0.0, 1.0}:
        raise ValueError(
            f"treatment indicator must be binary, found {sorted(set(np.unique(treated)))}"
        )
    if treated.sum() < 10 or (1 - treated).sum() < 10:
        raise InsufficientData(
            f"need >= 10 units in each treatment state, got {int(treated.sum())} treated "
            f"and {int((1 - treated).sum())} untreated"
        )

    design = frame[list(covariates)].to_numpy(dtype=float)
    # C=inf means no regularisation, which is what a propensity model wants: shrinking
    # coefficients biases the scores and therefore the weights. (`penalty=None` did the
    # same thing and was deprecated in scikit-learn 1.8.)
    model = LogisticRegression(C=np.inf, max_iter=1_000)
    model.fit(design, treated)
    scores = model.predict_proba(design)[:, 1]

    if scores.min() < 1e-6 or scores.max() > 1 - 1e-6:
        raise AssumptionViolation(
            f"positivity is violated: fitted propensities span "
            f"[{scores.min():.2e}, {1 - scores.max():.2e}] from the boundary. Some units "
            "had essentially no chance of the arm they received, so no reweighting can "
            "recover a valid comparison. Restrict the population to the region of overlap "
            "and say so, rather than weighting through it."
        )

    marginal = float(treated.mean())
    raw = np.where(treated == 1.0, 1.0 / scores, 1.0 / (1.0 - scores))
    weights = raw * np.where(treated == 1.0, marginal, 1.0 - marginal) if stabilised else raw

    return PropensityFit(
        scores=scores,
        treated=treated,
        covariates=tuple(covariates),
        weights=weights,
    )


def estimate_naive_difference(
    data: ExperimentData,
    estimand: Estimand,
    *,
    treatment_column: str | None = None,
    alpha: float = 0.05,
) -> EffectEstimate:
    """Difference in means, ignoring confounding entirely.

    The benchmark's lower bound. Under a selection regime this is expected to be badly
    biased, and including it is what makes the other estimators' performance mean
    something.
    """
    frame = data.frame
    if treatment_column is None:
        treated_mask = (frame[data.schema.variant_col] != data.control).to_numpy()
    else:
        treated_mask = frame[treatment_column].to_numpy(dtype=float) == 1.0

    y = frame[estimand.outcome].to_numpy(dtype=float)
    y_t, y_c = y[treated_mask], y[~treated_mask]
    if y_t.size < 2 or y_c.size < 2:
        raise InsufficientData(f"need >= 2 per group, got {y_t.size} and {y_c.size}")

    diff = float(y_t.mean() - y_c.mean())
    se = math.sqrt(y_t.var(ddof=1) / y_t.size + y_c.var(ddof=1) / y_c.size)
    z = stats.norm.isf(alpha / 2.0)

    return EffectEstimate(
        estimand=estimand,
        point=diff,
        ci=(diff - z * se, diff + z * se),
        ci_level=1.0 - alpha,
        se=se,
        p_value=float(2 * stats.norm.sf(abs(diff / se))) if se > 0 else 1.0,
        method="naive_difference",
        assumptions=(
            "NO adjustment for confounding is performed",
            "valid ONLY under randomised assignment; under selection on covariates this "
            "estimate is biased and the interval does not cover the true effect",
            "independent units",
        ),
        data_source=data.data_source,
        n_per_arm={"control": int(y_c.size), "treatment": int(y_t.size)},
        diagnostics={"mean_control": float(y_c.mean()), "mean_treatment": float(y_t.mean())},
    )


def estimate_ipw(
    data: ExperimentData,
    estimand: Estimand,
    covariates: Sequence[str],
    *,
    treatment_column: str | None = None,
    alpha: float = 0.05,
    stabilised: bool = True,
    trim: float | None = None,
) -> EffectEstimate:
    """Inverse-probability-weighted estimate of the ATE.

    Parameters
    ----------
    covariates
        The confounders being adjusted for. Conditional ignorability given these is
        assumed and cannot be checked -- it is recorded in ``assumptions``.
    trim
        If given, exclude units with propensity outside ``[trim, 1 - trim]``. **Off by
        default and never automatic.** Trimming changes the estimand from the ATE to the
        effect in the overlap region, so it must be a deliberate, recorded choice; the
        number of dropped units and the estimand change both land in ``assumptions``.

    Notes
    -----
    Standard errors use the weighted-mean sandwich form, which treats the propensity model
    as fixed. That understates uncertainty slightly, since fitting the propensity model is
    itself estimation. Stated rather than hidden; AIPW's influence-function variance is the
    better-behaved option.
    """
    fit = fit_propensity(data, covariates, treatment_column=treatment_column, stabilised=stabilised)
    y = data.frame[estimand.outcome].to_numpy(dtype=float)
    treated, weights, scores = fit.treated, fit.weights, fit.scores

    n_trimmed = 0
    if trim is not None:
        if not 0.0 < trim < 0.5:
            raise ValueError(f"trim must be in (0, 0.5), got {trim}")
        keep = (scores >= trim) & (scores <= 1.0 - trim)
        n_trimmed = int((~keep).sum())
        if keep.sum() < 20:
            raise InsufficientData(f"trimming at {trim} left only {int(keep.sum())} units")
        y, treated, weights, scores = y[keep], treated[keep], weights[keep], scores[keep]

    w_t, w_c = weights * treated, weights * (1 - treated)
    if w_t.sum() == 0 or w_c.sum() == 0:
        raise InsufficientData("one treatment group has zero total weight")

    mean_t = float(np.sum(w_t * y) / np.sum(w_t))
    mean_c = float(np.sum(w_c * y) / np.sum(w_c))
    point = mean_t - mean_c

    # Weighted variance of each arm's weighted mean.
    def weighted_var(values: np.ndarray, w: np.ndarray, mean: float) -> float:
        total = float(w.sum())
        if total == 0:
            return 0.0
        return float(np.sum(w**2 * (values - mean) ** 2) / total**2)

    se = math.sqrt(weighted_var(y, w_t, mean_t) + weighted_var(y, w_c, mean_c))
    z = stats.norm.isf(alpha / 2.0)

    if not fit.has_good_overlap:
        warnings.warn(
            f"poor propensity overlap: {fit.describe()}. The estimate may rest on very "
            "few units; inspect the weights before trusting it.",
            UserWarning,
            stacklevel=2,
        )

    assumptions = [
        f"conditional ignorability given {list(fit.covariates)} -- i.e. these are ALL the "
        "confounders. This is NOT testable from the data (R1.10)",
        "positivity: every unit had a non-zero chance of either arm",
        f"propensity model: logistic regression on {list(fit.covariates)}, correctly specified",
        f"{'stabilised' if stabilised else 'unstabilised'} inverse-probability weights",
        "standard errors treat the propensity model as fixed, so they slightly understate "
        "uncertainty; AIPW's influence-function variance is better behaved",
        "independent units",
    ]
    if trim is not None:
        assumptions.append(
            f"TRIMMED at propensity {trim}: {n_trimmed} unit(s) dropped. The estimand is "
            "now the effect in the overlap region, NOT the population ATE"
        )
    if not fit.has_good_overlap:
        assumptions.append(
            f"WARNING: poor overlap -- propensity range "
            f"[{fit.min_score:.4f}, {fit.max_score:.4f}], effective sample size "
            f"{fit.effective_sample_size:,.0f} of {fit.scores.size:,}"
        )
    if data.data_source is not DataSource.REAL:
        assumptions.append(f"data is {data.data_source} (R1.11)")

    return EffectEstimate(
        estimand=Estimand(
            outcome=estimand.outcome,
            treatment=estimand.treatment,
            target=EstimandTarget.ATE,
            population=("overlap region" if trim is not None else estimand.population),
            scale=estimand.scale,
        ),
        point=point,
        ci=(point - z * se, point + z * se),
        ci_level=1.0 - alpha,
        se=se,
        p_value=float(2 * stats.norm.sf(abs(point / se))) if se > 0 else 1.0,
        method="ipw",
        assumptions=tuple(assumptions),
        data_source=data.data_source,
        n_per_arm={
            "control": int((treated == 0).sum()),
            "treatment": int((treated == 1).sum()),
        },
        diagnostics={
            "min_propensity": fit.min_score,
            "max_propensity": fit.max_score,
            "max_weight": fit.max_weight,
            "effective_sample_size": fit.effective_sample_size,
            "ess_fraction": fit.ess_fraction,
            "n_trimmed": float(n_trimmed),
            "weighted_mean_control": mean_c,
            "weighted_mean_treatment": mean_t,
        },
    )
