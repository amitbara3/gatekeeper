"""The pre-registration spec -- the source of truth for an analysis (R1.2).

This is the keystone of the whole design. Pre-registration is what turns the
statistical rules from advice into something enforceable: the analysis *reads* the
spec, so it cannot silently deviate from the plan. A metric is not "primary" because
it moved; it is primary because the spec, committed before the first analysis run,
says so.

Changing a spec after seeing results is legitimate -- science is iterative -- but it
requires a **new spec file**, and the resulting analysis is labelled exploratory
rather than confirmatory. Editing a spec in place to match a result is the one thing
this file exists to prevent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gatekeeper.types import SpecViolation

__all__ = [
    "AnalysisMode",
    "ExperimentSpec",
    "OutlierRule",
    "load_spec",
]

AnalysisMode = Literal["confirmatory", "exploratory"]


class OutlierRule(BaseModel):
    """A pre-declared rule for handling extreme values (R1.6).

    Declared before analysis so that trimming is never a post-hoc reaction to seeing
    the data. Applied identically to every arm.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    method: Literal["none", "winsorize"] = "none"
    percentile: float | None = Field(
        default=None,
        description="Upper percentile for winsorising, e.g. 99.9. Required unless method='none'.",
    )

    @model_validator(mode="after")
    def _check_percentile(self) -> Self:
        if self.method == "none":
            if self.percentile is not None:
                raise ValueError(
                    f"outlier rule for {self.metric!r}: method='none' takes no percentile"
                )
        elif self.percentile is None:
            raise ValueError(
                f"outlier rule for {self.metric!r}: method={self.method!r} needs a percentile"
            )
        elif not 50.0 < self.percentile < 100.0:
            raise ValueError(
                f"outlier rule for {self.metric!r}: percentile must be in (50, 100), "
                f"got {self.percentile}"
            )
        return self

    def describe(self) -> str:
        if self.method == "none":
            return f"{self.metric}: no trimming; report the distribution as observed"
        return f"{self.metric}: winsorise at p{self.percentile:g}, applied identically to both arms"


class ExperimentSpec(BaseModel):
    """A pre-registered analysis plan.

    Every field that could otherwise be chosen after seeing results lives here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # -- identity --------------------------------------------------------------
    name: str
    dataset: str = Field(description="Dataset schema name, e.g. 'cookie_cats'.")
    registered_on: str = Field(description="ISO date the spec was committed, before analysis.")
    mode: AnalysisMode = Field(
        default="confirmatory",
        description=(
            "'confirmatory' for the pre-registered plan; 'exploratory' for any analysis "
            "written after seeing results. Never relabel a spec from one to the other."
        ),
    )
    notes: str = ""

    # -- metrics ---------------------------------------------------------------
    primary_metric: str
    direction: Literal["higher_is_better", "lower_is_better"]
    guardrail_metrics: tuple[str, ...] = ()
    declared_subgroups: tuple[str, ...] = Field(
        default=(),
        description="Subgroups permitted for heterogeneity analysis. Anything else is fishing (R1.9).",
    )

    # -- decision thresholds ---------------------------------------------------
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    power: float = Field(default=0.80, gt=0.0, lt=1.0)
    mde: float = Field(
        gt=0.0,
        description="Minimum detectable effect the test was designed for, absolute, in metric units.",
    )
    practical_threshold: float = Field(
        gt=0.0,
        description=(
            "Smallest effect magnitude that would change a decision, absolute, in metric "
            "units. The headline is judged against this, not against p < alpha (R1.4)."
        ),
    )

    # -- design ----------------------------------------------------------------
    expected_shares: dict[str, float] = Field(
        description="Intended allocation per arm; must sum to 1."
    )
    planned_n_per_arm: int | None = Field(default=None, gt=0)
    stopping_rule: Literal["fixed_horizon", "sequential"] = "fixed_horizon"
    srm_threshold: float = Field(default=0.0005, gt=0.0, lt=1.0)
    min_per_arm: int = Field(default=100, gt=0)

    # -- analysis choices ------------------------------------------------------
    multiplicity_method: Literal["benjamini_hochberg", "bonferroni", "none"] = "benjamini_hochberg"
    outlier_rules: tuple[OutlierRule, ...] = ()

    # -- validation ------------------------------------------------------------

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.primary_metric in self.guardrail_metrics:
            raise ValueError(
                f"primary metric {self.primary_metric!r} also appears in guardrail_metrics; "
                "a metric is one or the other"
            )
        if len(set(self.guardrail_metrics)) != len(self.guardrail_metrics):
            raise ValueError(f"duplicate guardrail metrics: {self.guardrail_metrics}")

        shares = sum(self.expected_shares.values())
        if abs(shares - 1.0) > 1e-9:
            raise ValueError(f"expected_shares must sum to 1, got {shares:.6f}")
        if any(v <= 0 for v in self.expected_shares.values()):
            raise ValueError(f"expected_shares must all be positive, got {self.expected_shares}")
        if len(self.expected_shares) < 2:
            raise ValueError("expected_shares must cover at least two arms")

        rule_metrics = [r.metric for r in self.outlier_rules]
        if len(set(rule_metrics)) != len(rule_metrics):
            raise ValueError(f"more than one outlier rule for the same metric: {rule_metrics}")

        if self.mde > self.practical_threshold:
            raise ValueError(
                f"mde ({self.mde}) exceeds practical_threshold ({self.practical_threshold}): "
                "the test cannot reliably detect effects as small as the smallest one that "
                "would change a decision, so a null result will be uninformative -- it will "
                "not distinguish 'no meaningful effect' from 'underpowered'. Either increase "
                "n until the MDE clears the threshold, or accept a larger threshold."
            )
        return self

    # -- derived ---------------------------------------------------------------

    @property
    def all_metrics(self) -> tuple[str, ...]:
        """Every declared metric, primary first. The set multiplicity corrects over."""
        return (self.primary_metric, *self.guardrail_metrics)

    @property
    def n_metrics(self) -> int:
        return len(self.all_metrics)

    def outlier_rule_for(self, metric: str) -> OutlierRule:
        """The declared rule for ``metric``, defaulting to no trimming."""
        for rule in self.outlier_rules:
            if rule.metric == metric:
                return rule
        return OutlierRule(metric=metric, method="none")

    # -- enforcement -----------------------------------------------------------

    def assert_metric_declared(self, metric: str) -> None:
        """Raise :class:`SpecViolation` if ``metric`` was not pre-registered.

        This is what stops the metric set from expanding silently as an analysis
        proceeds -- the precondition for multiplicity correction to mean anything
        (R1.8).
        """
        if metric not in self.all_metrics:
            raise SpecViolation(
                f"metric {metric!r} is not declared in spec {self.name!r} "
                f"(declared: {list(self.all_metrics)}). Analysing an undeclared metric "
                "and reporting it as a result is metric fishing. Either add it to a NEW "
                "spec file, or label this analysis exploratory."
            )

    def assert_primary(self, metric: str) -> None:
        """Raise unless ``metric`` is *the* pre-registered primary metric."""
        if metric != self.primary_metric:
            raise SpecViolation(
                f"{metric!r} is not the primary metric for spec {self.name!r} "
                f"(primary is {self.primary_metric!r}). The primary metric is fixed before "
                "analysis; promoting a metric because it moved is exactly the failure "
                "pre-registration prevents (R1.2)."
            )

    def assert_subgroup_declared(self, subgroup: str) -> None:
        """Raise unless ``subgroup`` was pre-declared (R1.9)."""
        if subgroup not in self.declared_subgroups:
            raise SpecViolation(
                f"subgroup {subgroup!r} is not declared in spec {self.name!r} "
                f"(declared: {list(self.declared_subgroups)}). Undeclared subgroup analysis "
                "is fishing; report the treatment x subgroup interaction instead."
            )

    def assert_single_look_allowed(self, look_number: int) -> None:
        """Raise if a fixed-horizon spec is being read more than once (R1.5).

        Repeated looks at accumulating data inflate the false-positive rate far above
        the nominal alpha. A fixed-horizon test may be read once, at the planned n.
        """
        if self.stopping_rule == "fixed_horizon" and look_number > 1:
            raise SpecViolation(
                f"spec {self.name!r} declares stopping_rule='fixed_horizon' but this is "
                f"look #{look_number}. Repeated looks inflate the false-positive rate well "
                "above alpha. Use a sequential spec with alpha-spending, or read once at "
                "the planned n (R1.5)."
            )

    def summary(self) -> str:
        arms = "  ".join(f"{k}={v:.1%}" for k, v in sorted(self.expected_shares.items()))
        return (
            f"spec {self.name!r} [{self.mode}] registered {self.registered_on}\n"
            f"  primary:    {self.primary_metric} ({self.direction})\n"
            f"  guardrails: {', '.join(self.guardrail_metrics) or '(none)'}\n"
            f"  alpha={self.alpha}  power={self.power}  mde={self.mde}  "
            f"practical={self.practical_threshold}\n"
            f"  allocation: {arms}  srm_threshold={self.srm_threshold:g}\n"
            f"  stopping:   {self.stopping_rule}  multiplicity={self.multiplicity_method}"
        )


def load_spec(path: str | Path) -> ExperimentSpec:
    """Load and validate a spec from YAML.

    Raises
    ------
    FileNotFoundError
        If the spec file does not exist.
    SpecViolation
        If the YAML is malformed or fails validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no spec at {p}")

    raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SpecViolation(f"spec at {p} must be a YAML mapping, got {type(raw).__name__}")

    try:
        return ExperimentSpec.model_validate(raw)
    except Exception as exc:
        raise SpecViolation(f"spec at {p} failed validation: {exc}") from exc
