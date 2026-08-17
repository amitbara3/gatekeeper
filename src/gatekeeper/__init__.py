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

from gatekeeper.checks.integrity import run_sanity_checks
from gatekeeper.checks.outliers import check_outlier_leverage, profile_metric
from gatekeeper.data.ingest import load_cookie_cats
from gatekeeper.data.schema import COOKIE_CATS, ExperimentData
from gatekeeper.data.synthetic import make_cookie_cats_like, make_null_experiment
from gatekeeper.design.srm import check_srm, srm_test
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

__version__ = "0.1.0"

__all__ = [
    "COOKIE_CATS",
    "AssumptionViolation",
    "DataSource",
    "Decision",
    "EffectEstimate",
    "Estimand",
    "EstimandTarget",
    "ExperimentData",
    "ExperimentSpec",
    "GatekeeperError",
    "InsufficientData",
    "PostTreatmentCovariateError",
    "SanityCheck",
    "SanityCheckFailure",
    "SanityReport",
    "Scale",
    "SchemaViolation",
    "SpecViolation",
    "__version__",
    "check_outlier_leverage",
    "check_srm",
    "load_cookie_cats",
    "load_spec",
    "make_cookie_cats_like",
    "make_null_experiment",
    "profile_metric",
    "run_sanity_checks",
    "srm_test",
]
