"""CATE estimation: S-, T-, and X-learners.

The average treatment effect answers "did it work?". The **conditional** average treatment
effect answers "for whom?" -- ``tau(x) = E[Y(1) - Y(0) | X = x]``. Three meta-learners,
each a different way of assembling ordinary regression models into a CATE estimate.

**S-learner** ("single") fits one model on ``(x, T)`` jointly and reads the difference
between predictions at ``T=1`` and ``T=0``. Simple, and *biased toward zero* by
construction: a regularised or tree-based learner given one treatment column among many
covariates will often ignore it, since dropping a weak feature costs little in prediction
loss. It is included precisely so that bias is visible rather than theoretical.

**T-learner** ("two") fits a separate model per arm and subtracts. No shrinkage of the
treatment signal, but it splits the data, so each model is fit on half as much -- painful
when one arm is small.

**X-learner** ("cross") is the refinement worth understanding. It fits per-arm outcome
models, then imputes each unit's *individual* treatment effect using the other arm's model,
then regresses those imputed effects on ``x``. Finally it blends the two resulting CATE
estimates using the propensity score as the weight. The blending is the point: where
treated units are scarce, the estimate leans on the model fit to the plentiful control
arm, and vice versa. That makes it markedly better than the T-learner under imbalance,
which is the situation it was designed for.

**The honest limit for this project.** Cookie Cats carries no pre-treatment covariates at
all -- only ``userid``. Genuine CATE estimation on it is impossible, and no amount of
machinery changes that (PRD §6). These learners are validated against a synthetic CATE
function that is known by construction. Saying so is the correct outcome for Phase 7, not
a shortfall.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from gatekeeper.data.schema import ExperimentData
from gatekeeper.types import InsufficientData

__all__ = ["CateEstimate", "LearnerKind", "estimate_cate"]

LearnerKind = Literal["s", "t", "x"]


@dataclass(frozen=True, slots=True)
class CateEstimate:
    """Per-unit conditional treatment effects."""

    tau: np.ndarray
    """Estimated ``tau(x_i)`` for every unit, in frame order."""
    learner: LearnerKind
    covariates: tuple[str, ...]
    ate: float
    """Mean of ``tau`` -- the implied ATE, useful as a sanity check against a direct
    estimate. A CATE model whose average disagrees with a well-behaved ATE estimator is
    telling you something is wrong."""

    @property
    def spread(self) -> float:
        """Standard deviation of the estimated effects.

        The crude measure of "how much heterogeneity did we find". A spread near zero
        means the model found none -- which may be the truth, or may be an S-learner
        shrinking the signal away.
        """
        return float(self.tau.std(ddof=1))

    def decile_effects(self) -> np.ndarray:
        """Mean estimated effect within each decile of estimated effect.

        The basis of an uplift curve: if the ranking is informative, the top decile's
        effect should be clearly larger than the bottom's.
        """
        order = np.argsort(self.tau)
        return np.array([float(chunk.mean()) for chunk in np.array_split(self.tau[order], 10)])

    def describe(self) -> str:
        deciles = self.decile_effects()
        return (
            f"{self.learner}-learner on {list(self.covariates)}: "
            f"ATE {self.ate:+.4f}, spread {self.spread:.4f}, "
            f"decile 1 {deciles[0]:+.4f} -> decile 10 {deciles[-1]:+.4f}"
        )


def _make_regressor(flexible: bool) -> Any:
    """A flexible learner for non-linear CATE, or plain OLS for the linear case.

    Returns ``Any`` because scikit-learn ships no shared protocol for "thing with fit and
    predict" that mypy can use, and inventing one here would be more ceremony than the
    two concrete classes justify.
    """
    if flexible:
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0
        )
    return LinearRegression()


def estimate_cate(
    data: ExperimentData,
    outcome: str,
    covariates: Sequence[str],
    *,
    learner: LearnerKind = "x",
    treatment_column: str | None = None,
    flexible: bool = True,
) -> CateEstimate:
    """Estimate per-unit conditional treatment effects.

    Parameters
    ----------
    data
        The experiment.
    outcome
        Outcome column.
    covariates
        Pre-treatment covariates. Checked against the schema's post-treatment flags --
        conditioning on a mediator is R1.7's error here as much as in CUPED.
    learner
        ``"s"``, ``"t"``, or ``"x"``. Defaults to the X-learner.
    treatment_column
        Received-treatment indicator; defaults to the variant column binarised.
    flexible
        Use gradient boosting for the nuisance models. ``False`` uses linear regression,
        which is the right choice when the true CATE is linear and n is small.

    Returns
    -------
    CateEstimate

    Raises
    ------
    PostTreatmentCovariateError
        If a covariate is measured after assignment.
    InsufficientData
        If either arm is too small to fit its models.
    """
    if not covariates:
        raise ValueError("CATE estimation needs at least one covariate")
    for name in covariates:
        data.assert_pre_treatment(name)

    frame = data.frame
    if treatment_column is None:
        treated = (frame[data.schema.variant_col] != data.control).to_numpy(dtype=float)
    else:
        treated = frame[treatment_column].to_numpy(dtype=float)

    x = frame[list(covariates)].to_numpy(dtype=float)
    y = frame[outcome].to_numpy(dtype=float)
    mask_t = treated == 1.0
    n_t, n_c = int(mask_t.sum()), int((~mask_t).sum())
    if n_t < 20 or n_c < 20:
        raise InsufficientData(
            f"each arm needs >= 20 units to fit a CATE model, got {n_t} and {n_c}"
        )

    if learner == "s":
        # One model on (x, T) jointly.
        design = np.column_stack([x, treated])
        model = _make_regressor(flexible).fit(design, y)
        tau = model.predict(np.column_stack([x, np.ones(x.shape[0])])) - model.predict(
            np.column_stack([x, np.zeros(x.shape[0])])
        )

    elif learner == "t":
        mu1 = _make_regressor(flexible).fit(x[mask_t], y[mask_t])
        mu0 = _make_regressor(flexible).fit(x[~mask_t], y[~mask_t])
        tau = mu1.predict(x) - mu0.predict(x)

    elif learner == "x":
        mu1 = _make_regressor(flexible).fit(x[mask_t], y[mask_t])
        mu0 = _make_regressor(flexible).fit(x[~mask_t], y[~mask_t])

        # Impute each unit's individual effect using the OTHER arm's model.
        imputed_treated = y[mask_t] - mu0.predict(x[mask_t])
        imputed_control = mu1.predict(x[~mask_t]) - y[~mask_t]

        # Regress the imputed effects on x, one model per arm.
        tau1 = _make_regressor(flexible).fit(x[mask_t], imputed_treated).predict(x)
        tau0 = _make_regressor(flexible).fit(x[~mask_t], imputed_control).predict(x)

        # Blend by propensity: lean on whichever arm has more units at this x.
        propensity = (
            LogisticRegression(C=np.inf, max_iter=1_000).fit(x, treated).predict_proba(x)[:, 1]
        )
        # Weighting tau0 by e(x) and tau1 by 1-e(x) is the standard X-learner blend: where
        # treatment is rare, e(x) is small, so weight shifts onto tau1 - the estimate built
        # from the few treated units' own outcomes against the well-fit control model.
        tau = propensity * tau0 + (1.0 - propensity) * tau1

    else:
        raise ValueError(f"unknown learner {learner!r}; expected 's', 't', or 'x'")

    return CateEstimate(
        tau=np.asarray(tau, dtype=float),
        learner=learner,
        covariates=tuple(covariates),
        ate=float(np.mean(tau)),
    )
