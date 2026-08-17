"""Score a set of estimates against a known true effect.

Four numbers, because no single one is sufficient:

**Bias** -- mean error. An estimator can be unbiased and useless (huge variance) so this
is necessary but not sufficient.

**Variance / RMSE** -- RMSE combines bias and variance, and is the number to compare
estimators on when you care about being close rather than merely centred.

**Coverage** -- how often the nominal 95% interval contains the truth. The one that
catches an estimator whose *point* is fine but whose *uncertainty* is wrong, which is a
common and dangerous failure: a confidently stated wrong interval is worse than a wide
honest one.

**Bias in SE units** -- ``|bias| / mean_se``. This is what makes bias interpretable. A
bias of 0.01 is negligible if the standard error is 0.5 and catastrophic if it is 0.001.
It also predicts coverage: bias much beyond 1 SE destroys coverage regardless of how well
the variance is estimated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gatekeeper.types import EffectEstimate, InsufficientData

__all__ = ["EstimatorScore", "score_estimates"]


@dataclass(frozen=True, slots=True)
class EstimatorScore:
    """Performance of one estimator against a known target."""

    n_replications: int
    true_value: float
    mean_estimate: float
    bias: float
    variance: float
    rmse: float
    coverage: float
    mean_se: float
    mean_ci_width: float
    alpha: float

    @property
    def bias_in_se_units(self) -> float:
        """``|bias| / mean_se`` -- the scale-free version of bias."""
        return math.inf if self.mean_se == 0 else abs(self.bias) / self.mean_se

    @property
    def coverage_is_nominal(self) -> bool:
        """Whether coverage is within Monte Carlo noise of ``1 - alpha``."""
        target = 1.0 - self.alpha
        se = math.sqrt(target * (1 - target) / self.n_replications)
        return abs(self.coverage - target) <= 3 * se

    @property
    def is_badly_biased(self) -> bool:
        """Bias beyond one standard error -- large enough to wreck coverage."""
        return self.bias_in_se_units > 1.0

    def describe(self) -> str:
        verdict = "biased" if self.is_badly_biased else "approximately unbiased"
        cov = "nominal" if self.coverage_is_nominal else "BROKEN"
        return (
            f"bias {self.bias:+.4f} ({self.bias_in_se_units:.2f} SE, {verdict})  "
            f"rmse {self.rmse:.4f}  coverage {self.coverage:.1%} ({cov})  "
            f"over {self.n_replications} replications"
        )


def score_estimates(
    estimates: Sequence[EffectEstimate],
    true_value: float,
    *,
    alpha: float = 0.05,
) -> EstimatorScore:
    """Score estimates against ``true_value``.

    Parameters
    ----------
    estimates
        Replicated estimates from independent draws of the same DGP.
    true_value
        The exact parameter. Not an estimate -- see
        :mod:`gatekeeper.causal.confounding` for why that distinction is load-bearing.
    alpha
        Nominal level of the intervals, for judging coverage.

    Raises
    ------
    InsufficientData
        If fewer than three replications.
    """
    if len(estimates) < 3:
        raise InsufficientData(
            f"need >= 3 replications to score an estimator, got {len(estimates)}"
        )

    points = np.array([e.point for e in estimates], dtype=float)
    covered = np.array([e.ci[0] <= true_value <= e.ci[1] for e in estimates], dtype=float)
    ses = np.array([0.0 if e.se is None else e.se for e in estimates], dtype=float)
    widths = np.array([e.ci[1] - e.ci[0] for e in estimates], dtype=float)

    bias = float(points.mean() - true_value)
    variance = float(points.var(ddof=1))

    return EstimatorScore(
        n_replications=len(estimates),
        true_value=true_value,
        mean_estimate=float(points.mean()),
        bias=bias,
        variance=variance,
        rmse=float(np.sqrt(np.mean((points - true_value) ** 2))),
        coverage=float(covered.mean()),
        mean_se=float(ses.mean()),
        mean_ci_width=float(widths.mean()),
        alpha=alpha,
    )
