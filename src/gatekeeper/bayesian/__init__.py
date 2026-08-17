"""Bayesian A/B testing.

Closed-form conjugate models only -- no MCMC (Architecture §1.1). The Beta prior is
conjugate to the Binomial likelihood, so the posterior is available exactly.
"""

from __future__ import annotations

from gatekeeper.bayesian.beta_binomial import (
    JEFFREYS_PRIOR,
    UNIFORM_PRIOR,
    BayesianComparison,
    BetaPosterior,
    BetaPrior,
    compare_beta_binomial,
    estimate_beta_binomial,
    prior_sensitivity,
    prob_b_beats_a,
)

__all__ = [
    "JEFFREYS_PRIOR",
    "UNIFORM_PRIOR",
    "BayesianComparison",
    "BetaPosterior",
    "BetaPrior",
    "compare_beta_binomial",
    "estimate_beta_binomial",
    "prior_sensitivity",
    "prob_b_beats_a",
]
