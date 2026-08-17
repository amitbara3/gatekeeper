"""Outcome regression and AIPW (doubly robust) estimation.

Two more ways to adjust for measured confounding, and the reason to prefer the second.

**Outcome regression** models ``E[Y | treatment, x]`` and reads the treatment coefficient.
Efficient when the outcome model is right, biased when it is wrong.

**AIPW** combines an outcome model with a propensity model and is **doubly robust**: it is
consistent if *either* model is correctly specified, not necessarily both. Two chances to
be right instead of one.

    tau = mean[ mu1(x) - mu0(x) + T(Y - mu1(x))/e(x) - (1-T)(Y - mu0(x))/(1 - e(x)) ]

The first term is the outcome-regression estimate; the rest is an IPW-weighted correction
built from the residuals. If the outcome model is right, the correction has mean zero and
does no harm. If the propensity model is right, the weighting fixes whatever the outcome
model got wrong. That is the whole trick.

**What double robustness does not buy.** It protects against *misspecification*, not
against *omission*. If a confounder is missing from both models, AIPW is exactly as biased
as everything else -- which the ``unobserved`` regime demonstrates rather than asserts.
Double robustness is a statement about functional form, and it is routinely oversold as
though it were a statement about confounding.

Standard errors come from the influence function, which accounts for estimating both
nuisance models -- unlike the plain IPW sandwich, which treats the propensity as fixed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from gatekeeper.causal.propensity import fit_propensity
from gatekeeper.data.schema import ExperimentData
from gatekeeper.types import (
    DataSource,
    EffectEstimate,
    Estimand,
    EstimandTarget,
    InsufficientData,
)

__all__ = ["estimate_aipw", "estimate_outcome_regression"]


def _treated_mask(data: ExperimentData, treatment_column: str | None) -> np.ndarray:
    if treatment_column is None:
        return (data.frame[data.schema.variant_col] != data.control).to_numpy(dtype=float)
    return data.frame[treatment_column].to_numpy(dtype=float)


def estimate_outcome_regression(
    data: ExperimentData,
    estimand: Estimand,
    covariates: Sequence[str],
    *,
    treatment_column: str | None = None,
    alpha: float = 0.05,
) -> EffectEstimate:
    """Regression adjustment: fit ``E[Y | T, x]`` and read the treatment coefficient.

    Uses separate models per arm (a T-learner) rather than one pooled model with a
    treatment dummy. The pooled version forces a common covariate slope across arms, which
    is an extra assumption bought for nothing when the data can support two fits.
    """
    if not covariates:
        raise ValueError("outcome regression needs at least one covariate")
    for name in covariates:
        data.assert_pre_treatment(name)

    treated = _treated_mask(data, treatment_column)
    x = data.frame[list(covariates)].to_numpy(dtype=float)
    y = data.frame[estimand.outcome].to_numpy(dtype=float)

    mask_t = treated == 1.0
    if mask_t.sum() < len(covariates) + 2 or (~mask_t).sum() < len(covariates) + 2:
        raise InsufficientData(
            f"each arm needs more units than covariates, got {int(mask_t.sum())} treated "
            f"and {int((~mask_t).sum())} control for {len(covariates)} covariate(s)"
        )

    model_t = LinearRegression().fit(x[mask_t], y[mask_t])
    model_c = LinearRegression().fit(x[~mask_t], y[~mask_t])

    # Average the predicted contrast over the whole sample: the ATE, not the ATT.
    mu1 = model_t.predict(x)
    mu0 = model_c.predict(x)
    point = float((mu1 - mu0).mean())

    # Standard error from the OLS coefficient covariance.
    #
    # The obvious-looking `(mu1 - mu0).std() / sqrt(n)` is WRONG and was the original
    # implementation here. That quantity measures how much the predicted contrast varies
    # ACROSS UNITS, not how uncertain the estimated average is. With a homogeneous
    # treatment effect the per-unit contrast is nearly constant, so it collapses toward
    # zero and produced 2.5% coverage with |bias|/se above 1000 in the benchmark.
    #
    # For a linear model the estimator is exactly a linear functional of the fitted
    # coefficients::
    #
    #     tau_hat = mean(mu1) - mean(mu0) = a'beta1 - a'beta0,   a = [1, mean(x)]
    #
    # and the arms are disjoint samples, so their coefficient covariances add::
    #
    #     Var(tau_hat) = a' V1 a + a' V0 a,   V_k = sigma_k^2 (Z_k' Z_k)^-1
    #
    # This conditions on the observed covariate distribution (treats `a` as fixed),
    # which is the standard convention for a sample ATE.
    a = np.concatenate([[1.0], x.mean(axis=0)])

    def coefficient_variance(design: np.ndarray, outcome: np.ndarray) -> float:
        z_mat = np.column_stack([np.ones(design.shape[0]), design])
        residuals = outcome - z_mat @ np.linalg.lstsq(z_mat, outcome, rcond=None)[0]
        dof = z_mat.shape[0] - z_mat.shape[1]
        if dof <= 0:
            raise InsufficientData("not enough units to estimate residual variance")
        sigma_sq = float(residuals @ residuals) / dof
        gram_inv = np.linalg.pinv(z_mat.T @ z_mat)
        return float(sigma_sq * (a @ gram_inv @ a))

    se = math.sqrt(
        coefficient_variance(x[mask_t], y[mask_t]) + coefficient_variance(x[~mask_t], y[~mask_t])
    )
    z = stats.norm.isf(alpha / 2.0)

    assumptions = [
        f"conditional ignorability given {list(covariates)} -- NOT testable (R1.10)",
        "the OUTCOME model is correctly specified (linear in these covariates). Unlike "
        "AIPW there is no second chance if it is not",
        "separate per-arm models, so covariate slopes may differ between arms",
        "the contrast is averaged over the full sample, targeting the ATE",
        "standard errors come from the OLS coefficient covariance, conditioning on the "
        "observed covariate distribution (the sample-ATE convention)",
        "homoskedastic residuals within each arm, which the OLS variance assumes",
        "independent units",
    ]
    if data.data_source is not DataSource.REAL:
        assumptions.append(f"data is {data.data_source} (R1.11)")

    return EffectEstimate(
        estimand=estimand,
        point=point,
        ci=(point - z * se, point + z * se),
        ci_level=1.0 - alpha,
        se=se,
        p_value=float(2 * stats.norm.sf(abs(point / se))) if se > 0 else 1.0,
        method="outcome_regression",
        assumptions=tuple(assumptions),
        data_source=data.data_source,
        n_per_arm={
            "control": int((~mask_t).sum()),
            "treatment": int(mask_t.sum()),
        },
        diagnostics={
            "mean_mu1": float(mu1.mean()),
            "mean_mu0": float(mu0.mean()),
            "n_covariates": float(len(covariates)),
        },
    )


def estimate_aipw(
    data: ExperimentData,
    estimand: Estimand,
    covariates: Sequence[str],
    *,
    treatment_column: str | None = None,
    alpha: float = 0.05,
) -> EffectEstimate:
    """Augmented IPW -- doubly robust estimation of the ATE.

    Consistent if **either** the outcome model or the propensity model is correct.

    Returns
    -------
    EffectEstimate
        With influence-function standard errors, which account for estimating both
        nuisance models.

    Notes
    -----
    Finite-sample coverage can fall below nominal when the propensity model produces
    extreme weights, because the influence function has heavy tails there. That is a
    documented property of the estimator rather than a defect in this implementation --
    and per R4.7 the way to tell those apart is that the shortfall must *shrink as n
    grows*, which the calibration tests check at two sample sizes.
    """
    if not covariates:
        raise ValueError("AIPW needs at least one covariate")

    fit = fit_propensity(data, covariates, treatment_column=treatment_column)
    treated = fit.treated
    scores = fit.scores
    x = data.frame[list(covariates)].to_numpy(dtype=float)
    y = data.frame[estimand.outcome].to_numpy(dtype=float)

    mask_t = treated == 1.0
    if mask_t.sum() < len(covariates) + 2 or (~mask_t).sum() < len(covariates) + 2:
        raise InsufficientData("each arm needs more units than covariates")

    mu1 = LinearRegression().fit(x[mask_t], y[mask_t]).predict(x)
    mu0 = LinearRegression().fit(x[~mask_t], y[~mask_t]).predict(x)

    # The doubly robust influence function, per unit.
    psi = mu1 - mu0 + treated * (y - mu1) / scores - (1.0 - treated) * (y - mu0) / (1.0 - scores)
    point = float(psi.mean())
    se = float(psi.std(ddof=1) / math.sqrt(psi.size))
    z = stats.norm.isf(alpha / 2.0)

    assumptions = [
        f"conditional ignorability given {list(covariates)} -- NOT testable, and double "
        "robustness does NOT relax it. If a confounder is missing from both models, AIPW "
        "is as biased as anything else (R1.10)",
        "DOUBLY ROBUST: consistent if EITHER the outcome model or the propensity model is "
        "correctly specified",
        "positivity: propensities bounded away from 0 and 1",
        "influence-function standard errors, accounting for both nuisance models",
        "independent units",
    ]
    if not fit.has_good_overlap:
        assumptions.append(
            f"WARNING: poor overlap -- propensity range "
            f"[{fit.min_score:.4f}, {fit.max_score:.4f}], ESS "
            f"{fit.effective_sample_size:,.0f}/{fit.scores.size:,}. The influence function "
            "has heavy tails here, so coverage may fall below nominal"
        )
    if data.data_source is not DataSource.REAL:
        assumptions.append(f"data is {data.data_source} (R1.11)")

    return EffectEstimate(
        estimand=Estimand(
            outcome=estimand.outcome,
            treatment=estimand.treatment,
            target=EstimandTarget.ATE,
            population=estimand.population,
            scale=estimand.scale,
        ),
        point=point,
        ci=(point - z * se, point + z * se),
        ci_level=1.0 - alpha,
        se=se,
        p_value=float(2 * stats.norm.sf(abs(point / se))) if se > 0 else 1.0,
        method="aipw",
        assumptions=tuple(assumptions),
        data_source=data.data_source,
        n_per_arm={
            "control": int((~mask_t).sum()),
            "treatment": int(mask_t.sum()),
        },
        diagnostics={
            "min_propensity": fit.min_score,
            "max_propensity": fit.max_score,
            "effective_sample_size": fit.effective_sample_size,
            "ess_fraction": fit.ess_fraction,
            "mean_mu1": float(mu1.mean()),
            "mean_mu0": float(mu0.mean()),
            "influence_sd": float(psi.std(ddof=1)),
        },
    )
