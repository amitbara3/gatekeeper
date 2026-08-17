"""The estimator benchmark -- the project's headline artifact (F5.8).

Runs each causal estimator across each confounding regime and scores it against a true
effect that is **exact by construction**, then reports bias, variance, RMSE, and interval
coverage.

**Predictions, written down before running (R2.2).** Architecture §5 states them and they
are asserted as tests, so the benchmark cannot quietly be reinterpreted after the fact:

1. On the **randomised** sample every estimator recovers tau, including the naive
   difference in means. An estimator failing here is broken, not challenged.
2. Under **selection** on an observed covariate, the naive difference is badly biased while
   IPW, outcome regression, and AIPW largely recover tau.
3. Under **unobserved** confounding **every** adjustment method fails. No amount of
   sophistication substitutes for having measured the confounder -- and the same estimators
   succeed when handed ``u``, which is what makes this a finding rather than a failure.

Point 3 is the one worth the whole exercise. It is easy to believe that doubly robust
methods are robust to confounding; they are robust to *misspecification*. The regime makes
the difference measurable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial

from gatekeeper.benchmark.scoring import EstimatorScore, score_estimates
from gatekeeper.causal.aipw import estimate_aipw, estimate_outcome_regression
from gatekeeper.causal.confounding import (
    CausalScenario,
    ConfoundingRegime,
    make_confounded,
    make_randomised,
)
from gatekeeper.causal.propensity import estimate_ipw, estimate_naive_difference
from gatekeeper.data.schema import ExperimentData
from gatekeeper.types import EffectEstimate, Estimand, GatekeeperError

__all__ = [
    "DEFAULT_ESTIMATORS",
    "BenchmarkResult",
    "EstimatorSpec",
    "run_benchmark",
]

EstimatorFn = Callable[[ExperimentData, Estimand, Sequence[str]], EffectEstimate]


@dataclass(frozen=True, slots=True)
class EstimatorSpec:
    """An estimator packaged for the harness.

    The uniform ``EffectEstimate`` return type is what makes this possible: the harness
    scores estimators without knowing anything about how they work (Architecture §2).
    """

    name: str
    fn: EstimatorFn
    uses_covariates: bool
    """``False`` for the naive difference, which adjusts for nothing."""


def _naive(data: ExperimentData, estimand: Estimand, _covariates: Sequence[str]) -> EffectEstimate:
    return estimate_naive_difference(data, estimand, treatment_column="received")


def _ipw(data: ExperimentData, estimand: Estimand, covariates: Sequence[str]) -> EffectEstimate:
    return estimate_ipw(data, estimand, covariates, treatment_column="received")


def _outcome(data: ExperimentData, estimand: Estimand, covariates: Sequence[str]) -> EffectEstimate:
    return estimate_outcome_regression(data, estimand, covariates, treatment_column="received")


def _aipw(data: ExperimentData, estimand: Estimand, covariates: Sequence[str]) -> EffectEstimate:
    return estimate_aipw(data, estimand, covariates, treatment_column="received")


def _build_randomised(n_units: int, true_ate: float, seed: int) -> CausalScenario:
    return make_randomised(n_units, true_ate=true_ate, seed=seed)


def _build_confounded(
    regime: ConfoundingRegime,
    n_units: int,
    true_ate: float,
    strength: float,
    seed: int,
) -> CausalScenario:
    return make_confounded(regime, n_units, true_ate=true_ate, strength=strength, seed=seed)


DEFAULT_ESTIMATORS: tuple[EstimatorSpec, ...] = (
    EstimatorSpec("naive_difference", _naive, uses_covariates=False),
    EstimatorSpec("ipw", _ipw, uses_covariates=True),
    EstimatorSpec("outcome_regression", _outcome, uses_covariates=True),
    EstimatorSpec("aipw", _aipw, uses_covariates=True),
)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Scores for every (regime, estimator) cell."""

    scores: dict[tuple[str, str], EstimatorScore]
    """Keyed by ``(regime, estimator_name)``."""
    covariates: tuple[str, ...]
    n_seeds: int
    n_units: int
    true_ate: float
    failures: dict[tuple[str, str], str] = field(default_factory=dict)
    """Cells where the estimator raised, with the reason. A refusal is a result: an
    estimator that declines to produce a number under total positivity failure is behaving
    better than one that produces a confident wrong one."""

    def score(self, regime: str, estimator: str) -> EstimatorScore:
        key = (regime, estimator)
        if key not in self.scores:
            raise KeyError(
                f"no score for {key}; available: {sorted(self.scores)}"
                + (f"; failed: {self.failures.get(key, '')}" if key in self.failures else "")
            )
        return self.scores[key]

    @property
    def regimes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r for r, _ in self.scores))

    @property
    def estimators(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e for _, e in self.scores))

    def render_table(self) -> str:
        """The benchmark table: bias and coverage per cell."""
        lines = [
            f"Estimator benchmark - true ATE {self.true_ate:+.4f}, "
            f"{self.n_seeds} seeds x {self.n_units:,} units, "
            f"covariates {list(self.covariates)}",
            "",
            f"{'regime':<16} {'estimator':<20} {'bias':>9} {'rmse':>9} {'coverage':>9} {'|bias|/se':>10}",
            "-" * 78,
        ]
        for regime in self.regimes:
            for estimator in self.estimators:
                key = (regime, estimator)
                if key in self.failures:
                    lines.append(f"{regime:<16} {estimator:<20} {'REFUSED':>9}")
                    continue
                s = self.scores[key]
                lines.append(
                    f"{regime:<16} {estimator:<20} {s.bias:>+9.4f} {s.rmse:>9.4f} "
                    f"{s.coverage:>8.1%} {s.bias_in_se_units:>10.2f}"
                )
        return "\n".join(lines)


def run_benchmark(
    *,
    n_units: int = 4_000,
    n_seeds: int = 40,
    true_ate: float = 1.0,
    strength: float = 1.2,
    covariates: Sequence[str] = ("x",),
    regimes: Sequence[ConfoundingRegime] = ("selection", "noncompliance", "unobserved"),
    estimators: Sequence[EstimatorSpec] = DEFAULT_ESTIMATORS,
    include_randomised: bool = True,
    alpha: float = 0.05,
) -> BenchmarkResult:
    """Score every estimator on every regime.

    Parameters
    ----------
    n_units
        Units generated per replication (before selection thins them).
    n_seeds
        Replications per cell. Bias precision goes as ``1/sqrt(n_seeds)``.
    true_ate
        The exact effect built into the DGP -- the scoring target.
    strength
        Confounding strength.
    covariates
        Covariates the estimators are **given**. Deliberately excludes ``u``: that
        withholding is what makes the ``unobserved`` regime a test of anything.
    regimes
        Which regimes to run.
    estimators
        Which estimators to score.
    include_randomised
        Also score an unconfounded sample, as the control condition.

    Returns
    -------
    BenchmarkResult
    """
    if n_seeds < 5:
        raise ValueError(f"n_seeds must be >= 5 for a meaningful score, got {n_seeds}")

    estimand = Estimand(outcome="y", treatment="received")

    # `partial` rather than lambdas with default arguments: the latter are a classic
    # late-binding trap in a loop, and mypy cannot infer their type either.
    builders: list[tuple[str, Callable[[int], CausalScenario]]] = []
    if include_randomised:
        builders.append(("randomised", partial(_build_randomised, n_units, true_ate)))
    for regime in regimes:
        builders.append((regime, partial(_build_confounded, regime, n_units, true_ate, strength)))

    scores: dict[tuple[str, str], EstimatorScore] = {}
    failures: dict[tuple[str, str], str] = {}

    for name, build in builders:
        per_estimator: dict[str, list[EffectEstimate]] = {s.name: [] for s in estimators}
        errors: dict[str, str] = {}

        for seed in range(n_seeds):
            scenario = build(seed)
            for spec in estimators:
                try:
                    per_estimator[spec.name].append(
                        spec.fn(scenario.data, estimand, list(covariates))
                    )
                except GatekeeperError as exc:
                    errors.setdefault(spec.name, f"{type(exc).__name__}: {exc}")

        for spec in estimators:
            estimates = per_estimator[spec.name]
            if len(estimates) < max(3, n_seeds // 2):
                failures[(name, spec.name)] = errors.get(
                    spec.name, f"only {len(estimates)}/{n_seeds} replications succeeded"
                )
                continue
            scores[(name, spec.name)] = score_estimates(estimates, true_ate, alpha=alpha)

    return BenchmarkResult(
        scores=scores,
        covariates=tuple(covariates),
        n_seeds=n_seeds,
        n_units=n_units,
        true_ate=true_ate,
        failures=failures,
    )
