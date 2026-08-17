"""Core types shared by every estimator, check, and report.

The central design commitment (Architecture §2): every analysis in this codebase --
frequentist or causal -- follows ``estimand -> estimator -> estimate``, and every
estimator returns an :class:`EffectEstimate`. That uniformity is what lets the
benchmark harness score arbitrary estimators, lets calibration tests be written once
and parametrised, and keeps method-specific branching out of the report layer.

``assumptions`` and ``data_source`` are required fields on :class:`EffectEstimate`.
An estimate therefore cannot be constructed without declaring what it assumes and
whether it came from real or synthetic data (Rules R2.4, R1.11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

__all__ = [
    "AssumptionViolation",
    "DataSource",
    "Decision",
    "EffectEstimate",
    "Estimand",
    "EstimandTarget",
    "GatekeeperError",
    "InsufficientData",
    "PostTreatmentCovariateError",
    "SanityCheck",
    "SanityCheckFailure",
    "SanityReport",
    "Scale",
    "SchemaViolation",
    "SpecViolation",
]


# ---------------------------------------------------------------------------
# Exceptions (Rules §5)
# ---------------------------------------------------------------------------


class GatekeeperError(Exception):
    """Base class for every error this package raises."""


class SchemaViolation(GatekeeperError):
    """Input data does not match the declared schema.

    Raised at ingest rather than coerced. Never ``fillna`` to make the code run.
    """


class SpecViolation(GatekeeperError):
    """An analysis deviates from the pre-registered experiment spec (R1.2)."""


class SanityCheckFailure(GatekeeperError):
    """Sanity checks failed and no override reason was supplied (R1.3).

    An SRM failure means the *instrumentation* is suspect. A clean p-value from
    broken assignment is not a weak result; it is not a result.
    """


class AssumptionViolation(GatekeeperError):
    """An estimator's identifying assumptions are provably violated (R1.10)."""


class PostTreatmentCovariateError(AssumptionViolation):
    """A post-treatment variable was passed where a pre-treatment one is required.

    The canonical case in this project is using ``sum_gamerounds`` as a CUPED
    covariate: it is measured *after* the player meets the gate, so it is a
    mediator. Adjusting for it biases the estimate while appearing to reduce
    variance -- the most tempting mistake available in this repo (R1.7).
    """


class InsufficientData(GatekeeperError):
    """Not enough data to compute the requested quantity."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataSource(StrEnum):
    """Provenance of the data behind an estimate (R1.11).

    Rendered as a visible badge everywhere results appear, so a figure lifted into
    a slide deck carries its own provenance (Design §7).
    """

    REAL = "real"
    SEMI_SYNTHETIC = "semi_synthetic"
    """Real data with deliberately injected confounding (the Phase 6 regimes)."""
    SYNTHETIC = "synthetic"


class Decision(StrEnum):
    """The outcome of a readout, judged against the spec's practical threshold."""

    SHIP = "ship"
    HOLD = "hold"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"
    """Sanity checks failed; metric results are not reported at all."""


class EstimandTarget(StrEnum):
    """Which causal quantity is being targeted.

    These are genuinely different numbers under heterogeneity or non-compliance.
    Comparing a LATE to an ATE as though they were the same quantity is the most
    common causal-inference error, which is why this is an explicit field.
    """

    ATE = "ATE"
    """Average treatment effect over the whole population."""
    ATT = "ATT"
    """Average effect among the treated."""
    LATE = "LATE"
    """Local average treatment effect -- compliers only."""
    CATE = "CATE"
    """Conditional average treatment effect -- a function of covariates."""


class Scale(StrEnum):
    ABSOLUTE = "absolute"
    """Difference in the metric's own units (percentage points for a rate)."""
    RELATIVE = "relative"
    """Ratio change: (treatment - control) / control."""


# ---------------------------------------------------------------------------
# Estimand
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Estimand:
    """What we are trying to estimate, stated *before* estimating it (R1.1).

    "The effect of the gate on retention" is not one number: it underspecifies the
    metric, the target quantity, the population, and the scale.

    Parameters
    ----------
    outcome
        Column name of the outcome metric.
    treatment
        Column name of the treatment indicator or variant label.
    target
        Which causal quantity (ATE, ATT, LATE, CATE).
    population
        Human-readable description of the population the estimate refers to.
    scale
        Absolute or relative.
    """

    outcome: str
    treatment: str
    target: EstimandTarget = EstimandTarget.ATE
    population: str = "all randomised units"
    scale: Scale = Scale.ABSOLUTE

    def __post_init__(self) -> None:
        if not self.outcome:
            raise ValueError("estimand.outcome must be a non-empty column name")
        if not self.treatment:
            raise ValueError("estimand.treatment must be a non-empty column name")

    def describe(self) -> str:
        """One-line human description, used as a chart subtitle (Design §3)."""
        return (
            f"{self.target} of {self.treatment} on {self.outcome} "
            f"({self.scale}, population: {self.population})"
        )


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SanityCheck:
    """Result of a single sanity check.

    Parameters
    ----------
    name
        Short identifier, e.g. ``"srm"`` or ``"duplicate_units"``.
    passed
        Whether the check passed.
    detail
        Human-readable explanation. Required -- a bare boolean is not actionable.
    statistic
        The test statistic, where one exists.
    p_value
        The p-value, where one exists.
    threshold
        The threshold the check was judged against, for reproducibility.
    """

    name: str
    passed: bool
    detail: str
    statistic: float | None = None
    p_value: float | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.detail:
            raise ValueError(f"sanity check {self.name!r} must carry a detail message")
        if self.p_value is not None and not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")


@dataclass(frozen=True, slots=True)
class SanityReport:
    """The gate between data and analysis (Architecture §2.1).

    This sits *between* ingest and estimation, not beside it: ``analyze()`` takes a
    ``SanityReport`` and refuses to proceed if it is failing. Making the safe path
    the default path is the whole design intent.
    """

    checks: tuple[SanityCheck, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> tuple[SanityCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def get(self, name: str) -> SanityCheck:
        """Return the named check, raising ``KeyError`` if it was not run."""
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(f"no sanity check named {name!r}; ran: {[c.name for c in self.checks]}")

    def raise_if_failed(self, override_reason: str | None = None) -> None:
        """Raise :class:`SanityCheckFailure` unless passing or explicitly overridden.

        Parameters
        ----------
        override_reason
            A recorded justification for proceeding despite failures. Stamped onto
            the resulting :class:`EffectEstimate` and rendered in the report, so an
            override is never silent (R1.3).
        """
        if self.passed or override_reason:
            return
        lines = "\n".join(f"  - {c.name}: {c.detail}" for c in self.failures)
        raise SanityCheckFailure(
            f"{len(self.failures)} sanity check(s) failed; results not reported:\n{lines}\n"
            "Assignment is suspect. Fix the instrumentation, or pass an "
            "override_reason to proceed on the record."
        )

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Sanity checks {status} ({len(self.checks) - len(self.failures)}/{len(self.checks)})"
        )


# ---------------------------------------------------------------------------
# EffectEstimate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    """A record of one completed estimation.

    Frozen and slotted: this is the record of a computation, not a mutable buffer.
    Every estimator in the package returns this type, whatever its method.

    Parameters
    ----------
    estimand
        What was being estimated -- declared before estimation (R1.1).
    point
        The point estimate, in the units implied by ``estimand.scale``.
    ci
        ``(lower, upper)`` interval. Never optional: a point estimate without an
        interval is not shippable output (Design §1).
    ci_level
        Coverage level of ``ci``, e.g. ``0.95``.
    se
        Standard error, where the method produces one.
    p_value
        Two-sided p-value, where the method produces one. Never reported alone
        (R1.4).
    method
        Identifier of the estimator, e.g. ``"two_proportion_z"``.
    assumptions
        What must hold for this estimate to be valid. **Required** -- an estimate
        cannot be constructed without it (R2.4, R1.10).
    data_source
        Real, semi-synthetic, or synthetic. **Required** (R1.11).
    n_per_arm
        Sample size per arm, keyed by variant label.
    diagnostics
        Method-specific numbers worth carrying forward (e.g. propensity overlap,
        achieved variance reduction, weight extremes).
    seed
        The RNG seed, where the method is stochastic (R4.2).
    override_reason
        Set when this estimate was produced despite failing sanity checks.
    """

    estimand: Estimand
    point: float
    ci: tuple[float, float]
    method: str
    assumptions: tuple[str, ...]
    data_source: DataSource
    n_per_arm: dict[str, int]
    ci_level: float = 0.95
    se: float | None = None
    p_value: float | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)
    seed: int | None = None
    override_reason: str | None = None

    def __post_init__(self) -> None:
        lo, hi = self.ci
        if lo > hi:
            raise ValueError(f"ci lower bound {lo} exceeds upper bound {hi}")
        if not 0.0 < self.ci_level < 1.0:
            raise ValueError(f"ci_level must be in (0, 1), got {self.ci_level}")
        if self.p_value is not None and not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")
        if self.se is not None and self.se < 0:
            raise ValueError(f"se must be non-negative, got {self.se}")
        if not self.assumptions:
            raise ValueError(
                f"method {self.method!r} produced an estimate with no declared "
                "assumptions; every estimate must state what it assumes (R2.4)"
            )
        if not self.method:
            raise ValueError("method must be a non-empty identifier")
        if not self.n_per_arm:
            raise ValueError("n_per_arm must record at least one arm")

    # -- interpretation helpers ------------------------------------------------

    @property
    def ci_width(self) -> float:
        return self.ci[1] - self.ci[0]

    @property
    def ci_excludes_zero(self) -> bool:
        """Whether the interval excludes no-effect. *Not* the same as mattering."""
        lo, hi = self.ci
        return lo > 0.0 or hi < 0.0

    @property
    def is_synthetic(self) -> bool:
        return self.data_source is not DataSource.REAL

    @property
    def n_total(self) -> int:
        return sum(self.n_per_arm.values())

    def is_practically_significant(self, threshold: float) -> bool:
        """Whether the whole interval lies beyond ``+/-threshold``.

        This is the strict reading, and the one this project uses: practical
        significance means we can *rule out* effects smaller than the threshold,
        not merely that the point estimate happened to exceed it. A point estimate
        of 2x the threshold with an interval straddling zero has not established
        anything (R1.4).

        Parameters
        ----------
        threshold
            Practical-significance threshold from the spec, as a positive
            magnitude in the same units as ``point``.
        """
        if threshold < 0:
            raise ValueError(f"threshold must be a non-negative magnitude, got {threshold}")
        lo, hi = self.ci
        return lo > threshold or hi < -threshold

    def excludes_effects_beyond(self, magnitude: float) -> bool:
        """Whether the interval rules out effects larger than ``magnitude``.

        The other half of an honest null result: "not significant" is only
        informative alongside what the test could actually rule out (R1.4).
        """
        if magnitude < 0:
            raise ValueError(f"magnitude must be non-negative, got {magnitude}")
        lo, hi = self.ci
        return lo > -magnitude and hi < magnitude

    def with_override(self, reason: str) -> Self:
        """Return a copy stamped with an override reason (R1.3)."""
        if not reason:
            raise ValueError("override reason must be non-empty")
        return type(self)(
            estimand=self.estimand,
            point=self.point,
            ci=self.ci,
            method=self.method,
            assumptions=self.assumptions,
            data_source=self.data_source,
            n_per_arm=dict(self.n_per_arm),
            ci_level=self.ci_level,
            se=self.se,
            p_value=self.p_value,
            diagnostics=dict(self.diagnostics),
            seed=self.seed,
            override_reason=reason,
        )

    def summary(self) -> str:
        """One-line rendering, in the number formats Design §3 mandates."""
        pct = round(self.ci_level * 100)
        p = "n/a" if self.p_value is None else _format_p(self.p_value)
        return (
            f"{self.point:+.4g} [{self.ci[0]:+.4g}, {self.ci[1]:+.4g}] ({pct}% CI)"
            f"  p={p}  method={self.method}  data={self.data_source}"
        )


def _format_p(p: float) -> str:
    """Format a p-value: 3 dp, or ``<0.001`` below that (Design §3)."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"
