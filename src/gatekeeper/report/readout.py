"""Turn estimates into a decision (F7.1, F7.4).

The decision is judged against the spec's **practical** threshold, not against
``p < 0.05`` (R1.4). A statistically significant 0.1pp retention change is not a
reason to ship anything.

**Guardrails are judged differently from the primary metric, on purpose.** The spec
carries one ``practical_threshold``, expressed in the primary metric's units --
retention proportion. Applying that same 0.01 to ``sum_gamerounds``, measured in
rounds, would be dimensionally meaningless: it would ask whether the effect on rounds
played exceeds 0.01 rounds, which every effect does. So:

- the **primary** metric is judged on *practical* significance (the whole interval
  beyond the threshold), and
- **guardrails** are judged on *statistical* significance after multiplicity
  correction, and a moved guardrail blocks a ship pending explanation.

That asymmetry is a real limitation of a single-threshold spec, not a design
preference; it is recorded as PRD open question O5. Inventing per-guardrail
thresholds here would be worse -- it would put numbers nobody chose into a decision
rule.
"""

from __future__ import annotations

from dataclasses import dataclass

from gatekeeper.frequentist.multiplicity import correct_spec_metrics
from gatekeeper.spec import ExperimentSpec
from gatekeeper.types import (
    DataSource,
    Decision,
    EffectEstimate,
    SanityReport,
    SpecViolation,
)

__all__ = ["MetricReadout", "Readout", "build_readout"]


@dataclass(frozen=True, slots=True)
class MetricReadout:
    """One metric's contribution to the decision."""

    metric: str
    estimate: EffectEstimate
    is_primary: bool
    adjusted_p: float | None
    """Multiplicity-adjusted p-value, or ``None`` if the estimator produced no p-value."""
    statistically_significant: bool
    """After adjustment. For guardrails this is the operative test."""
    practically_significant: bool
    """Whole interval beyond +/- the practical threshold. Primary metric only."""
    direction: str
    """``"improvement"``, ``"regression"``, or ``"unclear"``."""

    def describe(self) -> str:
        est = self.estimate
        label = "PRIMARY" if self.is_primary else "guardrail"
        p_text = "n/a" if self.adjusted_p is None else f"{self.adjusted_p:.4g}"
        return (
            f"[{label}] {self.metric}: {est.point:+.4g} "
            f"[{est.ci[0]:+.4g}, {est.ci[1]:+.4g}]  adj_p={p_text}  "
            f"direction={self.direction}"
        )


@dataclass(frozen=True, slots=True)
class Readout:
    """A complete, decision-bearing result for one experiment."""

    spec: ExperimentSpec
    sanity: SanityReport
    decision: Decision
    rationale: str
    metrics: tuple[MetricReadout, ...]
    data_source: DataSource
    override_reason: str | None = None

    @property
    def primary(self) -> MetricReadout:
        for m in self.metrics:
            if m.is_primary:
                return m
        raise ValueError("readout has no primary metric")

    @property
    def guardrails(self) -> tuple[MetricReadout, ...]:
        return tuple(m for m in self.metrics if not m.is_primary)

    @property
    def moved_guardrails(self) -> tuple[MetricReadout, ...]:
        """Guardrails that moved significantly and therefore need explanation."""
        return tuple(m for m in self.guardrails if m.statistically_significant)

    @property
    def is_blocked(self) -> bool:
        return self.decision is Decision.BLOCKED

    @property
    def is_synthetic(self) -> bool:
        return self.data_source is not DataSource.REAL

    def render_text(self) -> str:
        """Plain-text one-pager. The Design.md badge rules apply to the HTML version."""
        badge = {
            DataSource.REAL: "[real data]",
            DataSource.SEMI_SYNTHETIC: "[semi-synthetic - injected confounding]",
            DataSource.SYNTHETIC: "[SYNTHETIC DATA]",
        }[self.data_source]

        lines = [
            f"Gatekeeper readout - {self.spec.name}  {badge}",
            f"spec: {self.spec.name} ({self.spec.mode}, registered {self.spec.registered_on})",
            "",
            self.sanity.summary(),
        ]
        for check in self.sanity.checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{mark}] {check.name}")
        lines += ["", f"DECISION: {self.decision.value.upper()}", f"  {self.rationale}", ""]

        if not self.is_blocked:
            for m in self.metrics:
                lines.append(f"  {m.describe()}")
            lines.append("")
            lines.append(
                f"  practical threshold (primary only): +/-{self.spec.practical_threshold:g}"
            )
            lines.append(
                f"  multiplicity: {self.spec.multiplicity_method} over "
                f"{self.spec.n_metrics} declared metric(s)"
            )
        if self.override_reason:
            lines += ["", f"  OVERRIDE RECORDED: {self.override_reason}"]
        return "\n".join(lines)


def _classify_direction(estimate: EffectEstimate, higher_is_better: bool) -> str:
    """Whether a metric moved for better, for worse, or indeterminately."""
    if not estimate.ci_excludes_zero:
        return "unclear"
    improved = estimate.point > 0 if higher_is_better else estimate.point < 0
    return "improvement" if improved else "regression"


def build_readout(
    spec: ExperimentSpec,
    sanity: SanityReport,
    estimates: dict[str, EffectEstimate],
    *,
    override_reason: str | None = None,
) -> Readout:
    """Assemble estimates into a decision.

    Parameters
    ----------
    spec
        The pre-registered plan. Supplies the primary metric, the practical
        threshold, alpha, and the multiplicity method -- none of which this function
        may choose for itself (R1.2).
    sanity
        The gate's report. A failing report yields ``Decision.BLOCKED`` and no metric
        results, unless ``override_reason`` is given (R1.3).
    estimates
        One :class:`EffectEstimate` per declared metric, keyed by metric name. Must
        cover exactly ``spec.all_metrics`` -- a family that grows or shrinks makes the
        multiplicity correction meaningless (R1.8).
    override_reason
        Recorded justification for reporting despite failing sanity checks. Stamped
        onto the readout and rendered in the output.

    Returns
    -------
    Readout

    Raises
    ------
    SpecViolation
        If ``estimates`` does not exactly match the spec's declared metric family.
    """
    declared = set(spec.all_metrics)
    supplied = set(estimates)
    if missing := declared - supplied:
        raise SpecViolation(
            f"no estimate supplied for declared metric(s) {sorted(missing)}; the whole "
            "pre-registered family must be analysed or the correction is invalid (R1.8)"
        )
    if extra := supplied - declared:
        raise SpecViolation(
            f"estimate(s) supplied for undeclared metric(s) {sorted(extra)}; add them to "
            "a NEW spec or label the analysis exploratory (R1.2)"
        )

    sources = {e.data_source for e in estimates.values()}
    if len(sources) > 1:
        raise ValueError(
            f"estimates mix data provenance {sorted(s.value for s in sources)}; a single "
            "readout must not blend real and synthetic results (R1.11)"
        )
    data_source = sources.pop()

    # --- blocked? ------------------------------------------------------------
    if not sanity.passed and not override_reason:
        failures = ", ".join(c.name for c in sanity.failures)
        return Readout(
            spec=spec,
            sanity=sanity,
            decision=Decision.BLOCKED,
            rationale=(
                f"{len(sanity.failures)} sanity check(s) failed ({failures}). Assignment "
                "or logging is suspect, so metric results are not reported. Fix the "
                "instrumentation, or re-run with a recorded override reason."
            ),
            metrics=(),
            data_source=data_source,
            override_reason=None,
        )

    # --- multiplicity across the declared family -----------------------------
    # An estimator with no p-value (a bootstrap, say) enters the family as p=1, so it
    # neither claims significance nor changes anyone else's divisor. Written as a loop
    # rather than a comprehension so the None-narrowing is explicit.
    p_by_metric: dict[str, float] = {}
    for metric_name in spec.all_metrics:
        raw_p = estimates[metric_name].p_value
        p_by_metric[metric_name] = 1.0 if raw_p is None else raw_p
    corrected = correct_spec_metrics(
        p_by_metric,
        spec.all_metrics,
        alpha=spec.alpha,
        method=spec.multiplicity_method,
    )

    higher_is_better = spec.direction == "higher_is_better"
    readouts: list[MetricReadout] = []
    for metric in spec.all_metrics:
        estimate = estimates[metric]
        adjusted_p, rejected = corrected[metric]
        is_primary = metric == spec.primary_metric
        readouts.append(
            MetricReadout(
                metric=metric,
                estimate=estimate,
                is_primary=is_primary,
                adjusted_p=None if estimate.p_value is None else adjusted_p,
                statistically_significant=rejected and estimate.p_value is not None,
                practically_significant=(
                    is_primary and estimate.is_practically_significant(spec.practical_threshold)
                ),
                # NOTE: `direction` uses the primary metric's higher_is_better
                # convention for every metric, which is only strictly meaningful for
                # the primary one -- the spec declares no per-guardrail direction. For
                # guardrails the operative signal is `statistically_significant`
                # ("moved at all"), and the decision rule uses only that. Reported here
                # as a hint for a human reading the output, not as a decision input.
                direction=_classify_direction(estimate, higher_is_better),
            )
        )

    primary = next(r for r in readouts if r.is_primary)
    moved = [r for r in readouts if not r.is_primary and r.statistically_significant]
    threshold = spec.practical_threshold

    # --- decide --------------------------------------------------------------
    if primary.practically_significant and primary.direction == "regression":
        decision = Decision.HOLD
        rationale = (
            f"{primary.metric} regressed by a practically significant margin: "
            f"{primary.estimate.point:+.4g} with a {int(primary.estimate.ci_level * 100)}% "
            f"interval of [{primary.estimate.ci[0]:+.4g}, {primary.estimate.ci[1]:+.4g}], "
            f"entirely beyond the {threshold:g} threshold in the harmful direction."
        )
    elif primary.practically_significant and primary.direction == "improvement":
        if moved:
            names = ", ".join(r.metric for r in moved)
            decision = Decision.HOLD
            rationale = (
                f"{primary.metric} improved by a practically significant margin "
                f"({primary.estimate.point:+.4g}), but guardrail(s) {names} also moved "
                "significantly after correction. A guardrail that moves without "
                "explanation is exactly what guardrails exist to catch; explain it "
                "before shipping."
            )
        else:
            decision = Decision.SHIP
            rationale = (
                f"{primary.metric} improved by {primary.estimate.point:+.4g} with a "
                f"{int(primary.estimate.ci_level * 100)}% interval of "
                f"[{primary.estimate.ci[0]:+.4g}, {primary.estimate.ci[1]:+.4g}], entirely "
                f"beyond the {threshold:g} practical threshold, and no guardrail moved "
                "significantly."
            )
    else:
        decision = Decision.INCONCLUSIVE
        ruled_out = primary.estimate.excludes_effects_beyond(threshold)
        if ruled_out:
            detail = (
                f"The interval lies entirely inside +/-{threshold:g}, so an effect large "
                "enough to matter has been ruled out. This is an informative null, not a "
                "failure to measure."
            )
        else:
            detail = (
                f"The interval is not entirely beyond +/-{threshold:g}, and neither is it "
                "entirely inside it -- so a practically important effect can be neither "
                "confirmed nor excluded. The test lacked the precision to answer its own "
                "question at this threshold."
            )
        rationale = (
            f"{primary.metric} moved {primary.estimate.point:+.4g} with a "
            f"{int(primary.estimate.ci_level * 100)}% interval of "
            f"[{primary.estimate.ci[0]:+.4g}, {primary.estimate.ci[1]:+.4g}]. {detail}"
        )
        if moved:
            names = ", ".join(r.metric for r in moved)
            rationale += f" Note: guardrail(s) {names} moved significantly."

    return Readout(
        spec=spec,
        sanity=sanity,
        decision=decision,
        rationale=rationale,
        metrics=tuple(readouts),
        data_source=data_source,
        override_reason=override_reason,
    )
