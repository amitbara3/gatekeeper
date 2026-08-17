"""Causal inference for when randomisation is absent or broken.

The confounding simulator plus the estimators the benchmark scores. IV, DiD, and
Rosenbaum bounds are deliberately out of scope for now -- Cookie Cats has neither a time
dimension nor covariates, so those would be synthetic-only demonstrations with low
learning yield relative to cost. E-values give most of the sensitivity-analysis intuition
for a fraction of the work.
"""

from __future__ import annotations

from gatekeeper.causal.aipw import estimate_aipw, estimate_outcome_regression
from gatekeeper.causal.confounding import (
    CONFOUNDED_SCHEMA,
    CausalScenario,
    ConfoundingRegime,
    covariate_imbalance,
    make_confounded,
    make_randomised,
)
from gatekeeper.causal.propensity import (
    PropensityFit,
    estimate_ipw,
    estimate_naive_difference,
    fit_propensity,
)
from gatekeeper.causal.sensitivity import EValue, e_value, e_value_from_difference

__all__ = [
    "CONFOUNDED_SCHEMA",
    "CausalScenario",
    "ConfoundingRegime",
    "EValue",
    "PropensityFit",
    "covariate_imbalance",
    "e_value",
    "e_value_from_difference",
    "estimate_aipw",
    "estimate_ipw",
    "estimate_naive_difference",
    "estimate_outcome_regression",
    "fit_propensity",
    "make_confounded",
    "make_randomised",
]
