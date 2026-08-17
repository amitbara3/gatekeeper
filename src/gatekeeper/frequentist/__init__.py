"""Frequentist estimators for the everyday readout.

Every estimator here comes in two layers: a **core** function on plain arrays or
counts, and an :class:`~gatekeeper.types.EffectEstimate`-returning wrapper that takes
an :class:`~gatekeeper.data.schema.ExperimentData`. The split lets the calibration
suite run tens of thousands of simulations against the core without paying for frame
construction on every draw, and keeps the statistics readable in isolation from the
data plumbing.
"""

from __future__ import annotations

from gatekeeper.frequentist.bootstrap import (
    BootstrapResult,
    bootstrap_mean_difference,
    bootstrap_statistic,
    estimate_bootstrap,
)
from gatekeeper.frequentist.means import MeansResult, estimate_welch, welch_test
from gatekeeper.frequentist.multiplicity import (
    CorrectionResult,
    correct,
    correct_spec_metrics,
)
from gatekeeper.frequentist.proportions import (
    ProportionResult,
    estimate_two_proportion,
    two_proportion_test,
)
from gatekeeper.frequentist.ratio import (
    RatioDifference,
    RatioEstimate,
    ratio_difference,
    ratio_variance,
)

__all__ = [
    "BootstrapResult",
    "CorrectionResult",
    "MeansResult",
    "ProportionResult",
    "RatioDifference",
    "RatioEstimate",
    "bootstrap_mean_difference",
    "bootstrap_statistic",
    "correct",
    "correct_spec_metrics",
    "estimate_bootstrap",
    "estimate_two_proportion",
    "estimate_welch",
    "ratio_difference",
    "ratio_variance",
    "two_proportion_test",
    "welch_test",
]
