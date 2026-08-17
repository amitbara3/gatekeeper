"""Outlier profiling. Reports, never trims (R1.6).

Nothing in this module removes data. Discovering an extreme value, dropping it, and
re-running is p-hacking however reasonable it feels -- so the outlier rule is
declared in the spec *before* analysis, and trimming is an explicit caller decision
driven by that rule.

What this module does provide is the evidence needed to write a sensible rule in the
first place, and a gate that fires when a metric's mean is so dominated by single
observations that a mean-based test is not defensible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gatekeeper.data.schema import ExperimentData
from gatekeeper.spec import OutlierRule
from gatekeeper.types import InsufficientData, SanityCheck

__all__ = ["OutlierProfile", "check_outlier_leverage", "profile_metric"]

_PERCENTILES = (50.0, 90.0, 99.0, 99.9)


@dataclass(frozen=True, slots=True)
class OutlierProfile:
    """Distributional summary of one metric in one arm.

    Attributes
    ----------
    metric, variant
        What was profiled.
    n
        Number of observations.
    mean, median, std, minimum, maximum
        Standard summaries.
    percentiles
        Mapping of percentile to value, for the percentiles in ``_PERCENTILES``.
    top_share
        Share of the metric's total contributed by the largest 0.1% of units.
    max_leverage
        Relative shift in the arm mean caused by dropping the single largest
        observation. ``0.02`` means one unit moves the mean by 2%.
    """

    metric: str
    variant: str
    n: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    percentiles: dict[float, float]
    top_share: float
    max_leverage: float

    @property
    def skew_ratio(self) -> float:
        """``mean / median`` -- a crude but legible skew signal.

        Well above 1 means the mean is being pulled by a tail, and the median or a
        rank-based test is likely the more honest summary (R1.12).
        """
        return float("inf") if self.median == 0 else self.mean / self.median

    def describe(self) -> str:
        pcts = "  ".join(f"p{p:g}={self.percentiles[p]:,.1f}" for p in sorted(self.percentiles))
        return (
            f"{self.metric} [{self.variant}]  n={self.n:,}\n"
            f"  mean={self.mean:,.3f}  median={self.median:,.1f}  std={self.std:,.3f}  "
            f"mean/median={self.skew_ratio:,.2f}\n"
            f"  {pcts}  max={self.maximum:,.0f}\n"
            f"  top 0.1% hold {self.top_share:.2%} of the total; "
            f"largest single unit moves the mean by {self.max_leverage:.3%}"
        )


def _reject_binary_metric(data: ExperimentData, metric: str) -> None:
    """Refuse to profile a Bernoulli metric.

    Every quantity this module computes -- tail percentiles, top-share, the leverage
    of the largest observation -- presupposes a metric with a *magnitude*. For a
    binary metric none of them mean anything: the "largest observation" is 1, the
    "tail" is just the event rate, and leverage is mechanically ~1/n.

    The failure mode this guards against is not a crash but a plausible-looking
    number. Pointed at ``retention_7`` the check would compute a tiny leverage,
    report "no single unit dominates the mean", and pass -- an answer that is
    arithmetically true and analytically empty. Silently returning a confident number
    for an inappropriate input is precisely the class of error this project exists to
    catch, so it raises instead.
    """
    try:
        kind = data.schema.column(metric).kind
    except KeyError:
        return  # not a schema-declared column; nothing to assert
    if kind == "bool":
        raise ValueError(
            f"{metric!r} is a binary metric, so outlier profiling does not apply: its "
            "percentiles, top-share, and leverage are all artefacts of the event rate "
            "rather than of a tail. Use a two-proportion test for binary metrics, and "
            "reserve this module for magnitude metrics such as sum_gamerounds."
        )


def profile_metric(data: ExperimentData, metric: str, variant: str) -> OutlierProfile:
    """Summarise the distribution of ``metric`` within one arm.

    Parameters
    ----------
    data
        The experiment.
    metric
        Column to profile. Must be a magnitude metric, not a binary one.
    variant
        Arm label.

    Raises
    ------
    ValueError
        If ``metric`` is declared boolean -- outlier concepts do not apply to a
        Bernoulli metric.
    InsufficientData
        If the arm has fewer than two observations.
    """
    _reject_binary_metric(data, metric)
    values = data.outcome(metric, variant)
    n = values.size
    if n < 2:
        raise InsufficientData(
            f"cannot profile {metric!r} for arm {variant!r}: need >= 2 observations, got {n}"
        )

    total = float(values.sum())
    mean = float(values.mean())

    # Leverage of the single most extreme observation on the arm mean.
    ordered = np.sort(values)
    without_max = ordered[:-1]
    max_leverage = 0.0 if mean == 0 else abs(float(without_max.mean()) - mean) / abs(mean)

    # Share of the total held by the top 0.1% of units (at least one unit).
    k = max(1, int(np.ceil(n * 0.001)))
    top_share = 0.0 if total == 0 else float(ordered[-k:].sum()) / total

    return OutlierProfile(
        metric=metric,
        variant=variant,
        n=n,
        mean=mean,
        median=float(np.median(values)),
        std=float(values.std(ddof=1)),
        minimum=float(ordered[0]),
        maximum=float(ordered[-1]),
        percentiles={p: float(np.percentile(values, p)) for p in _PERCENTILES},
        top_share=top_share,
        max_leverage=max_leverage,
    )


def check_outlier_leverage(
    data: ExperimentData,
    metric: str,
    *,
    leverage_threshold: float = 0.01,
    declared_rule: OutlierRule | None = None,
) -> SanityCheck:
    """Check that a metric's sensitivity to extreme values was planned for.

    This does not detect "outliers" in the abstract -- it detects whether a
    mean-based test on this metric is *defensible*, and whether the spec anticipated
    the problem. If one unit out of tens of thousands moves an arm mean by more than
    ``leverage_threshold``, the mean is reporting that unit rather than the
    population (R1.12).

    The check therefore fails on **high leverage with no pre-declared rule**, not on
    high leverage as such. A heavy tail is a property of the metric, not a defect: a
    game's rounds-played distribution is genuinely lognormal-ish, and blocking a
    readout for that would be wrong. What must not happen is meeting the tail for the
    first time *during* analysis and improvising -- which is why the pass condition
    is "you decided in advance".

    Parameters
    ----------
    data
        The experiment.
    metric
        Column to check.
    leverage_threshold
        Maximum relative shift in an arm mean, from dropping its single largest
        observation, that needs no pre-declared rule. Default 1%.
    declared_rule
        The spec's rule for this metric, from
        :meth:`~gatekeeper.spec.ExperimentSpec.outlier_rule_for`. A rule with
        ``method != "none"`` satisfies the check even under high leverage.

    Returns
    -------
    SanityCheck
        **Never modifies the data** -- it reports (R1.6).

    Raises
    ------
    ValueError
        If ``leverage_threshold`` is non-positive, or ``metric`` is binary.
    """
    if leverage_threshold <= 0:
        raise ValueError(f"leverage_threshold must be positive, got {leverage_threshold}")
    _reject_binary_metric(data, metric)

    profiles = [profile_metric(data, metric, v) for v in data.variants]
    worst = max(profiles, key=lambda p: p.max_leverage)
    high_leverage = worst.max_leverage > leverage_threshold
    has_rule = declared_rule is not None and declared_rule.method != "none"

    shared = (
        f"worst arm {worst.variant!r}: one unit shifts the mean by "
        f"{worst.max_leverage:.3%} (max={worst.maximum:,.0f}, mean={worst.mean:,.2f}, "
        f"median={worst.median:,.1f}, mean/median={worst.skew_ratio:,.2f})"
    )

    if not high_leverage:
        passed, detail = True, f"{metric}: no single unit dominates the mean; {shared}."
    elif has_rule:
        assert declared_rule is not None  # narrowed by has_rule
        passed = True
        detail = (
            f"{metric}: heavy-tailed as expected -- {shared}, above the "
            f"{leverage_threshold:.1%} threshold. The spec anticipated this and declares: "
            f"{declared_rule.describe()}. Apply it identically to both arms and report "
            "results with and without it."
        )
    else:
        passed = False
        detail = (
            f"{metric}: mean is dominated by extreme values and NO rule was pre-declared -- "
            f"{shared}, above the {leverage_threshold:.1%} threshold. A mean-based test here "
            "reports the tail, not the population. Add an outlier rule to the spec, or use "
            "the bootstrap/rank alternative. Do not discover-and-drop (R1.6)."
        )

    return SanityCheck(
        name=f"outlier_leverage:{metric}",
        passed=passed,
        detail=detail,
        statistic=worst.max_leverage,
        threshold=leverage_threshold,
    )
