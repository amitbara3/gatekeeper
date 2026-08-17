"""Gatekeeper -- an experimentation & causal inference workbench.

See PRD.md for scope, Architecture.md for design, and Rules.md for the statistical
and code constraints this package is built under.

The core flow (Architecture §2)::

    spec = load_spec("specs/cookie_cats_gate.yaml")
    data = load_cookie_cats()
    report = run_sanity_checks(data, expected_shares=spec.expected_shares,
                               srm_threshold=spec.srm_threshold)
    report.raise_if_failed()          # the gate -- fails loudly, overrides recorded
    # ... estimators land in Phase 2
"""

from __future__ import annotations

from gatekeeper.bayesian.beta_binomial import (
    JEFFREYS_PRIOR,
    UNIFORM_PRIOR,
    BetaPrior,
    compare_beta_binomial,
    estimate_beta_binomial,
    prior_sensitivity,
)
from gatekeeper.checks.integrity import run_sanity_checks
from gatekeeper.checks.outliers import check_outlier_leverage, profile_metric
from gatekeeper.data.ingest import load_cookie_cats
from gatekeeper.data.schema import COOKIE_CATS, ExperimentData
from gatekeeper.data.synthetic import (
    make_cookie_cats_like,
    make_null_experiment,
    make_pre_period_experiment,
)
from gatekeeper.design.power import (
    duration_days,
    mde_means,
    mde_two_proportion,
    power_means,
    power_two_proportion,
    sample_size_means,
    sample_size_two_proportion,
)
from gatekeeper.design.srm import check_srm, srm_test
from gatekeeper.frequentist.bootstrap import bootstrap_mean_difference, estimate_bootstrap
from gatekeeper.frequentist.means import estimate_welch, welch_test
from gatekeeper.frequentist.multiplicity import correct, correct_spec_metrics
from gatekeeper.frequentist.proportions import estimate_two_proportion, two_proportion_test
from gatekeeper.frequentist.ratio import ratio_difference, ratio_variance
from gatekeeper.report.readout import MetricReadout, Readout, build_readout
from gatekeeper.sequential.always_valid import (
    always_valid_interval,
    always_valid_p_value,
    sequential_p_values,
    suggest_tau,
)
from gatekeeper.sequential.peeking import simulate_peeking
from gatekeeper.spec import ExperimentSpec, load_spec
from gatekeeper.types import (
    AssumptionViolation,
    DataSource,
    Decision,
    EffectEstimate,
    Estimand,
    EstimandTarget,
    GatekeeperError,
    InsufficientData,
    PostTreatmentCovariateError,
    SanityCheck,
    SanityCheckFailure,
    SanityReport,
    Scale,
    SchemaViolation,
    SpecViolation,
)
from gatekeeper.variance.cuped import cuped_adjust, cuped_theta, estimate_cuped

__version__ = "0.1.0"

__all__ = [
    "COOKIE_CATS",
    "JEFFREYS_PRIOR",
    "UNIFORM_PRIOR",
    "AssumptionViolation",
    "BetaPrior",
    "DataSource",
    "Decision",
    "EffectEstimate",
    "Estimand",
    "EstimandTarget",
    "ExperimentData",
    "ExperimentSpec",
    "GatekeeperError",
    "InsufficientData",
    "MetricReadout",
    "PostTreatmentCovariateError",
    "Readout",
    "SanityCheck",
    "SanityCheckFailure",
    "SanityReport",
    "Scale",
    "SchemaViolation",
    "SpecViolation",
    "__version__",
    "always_valid_interval",
    "always_valid_p_value",
    "bootstrap_mean_difference",
    "build_readout",
    "check_outlier_leverage",
    "check_srm",
    "compare_beta_binomial",
    "correct",
    "correct_spec_metrics",
    "cuped_adjust",
    "cuped_theta",
    "duration_days",
    "estimate_beta_binomial",
    "estimate_bootstrap",
    "estimate_cuped",
    "estimate_two_proportion",
    "estimate_welch",
    "load_cookie_cats",
    "load_spec",
    "make_cookie_cats_like",
    "make_null_experiment",
    "make_pre_period_experiment",
    "mde_means",
    "mde_two_proportion",
    "power_means",
    "power_two_proportion",
    "prior_sensitivity",
    "profile_metric",
    "ratio_difference",
    "ratio_variance",
    "run_sanity_checks",
    "sample_size_means",
    "sample_size_two_proportion",
    "sequential_p_values",
    "simulate_peeking",
    "srm_test",
    "suggest_tau",
    "two_proportion_test",
    "welch_test",
]
