"""Multiple-comparison correction across a declared metric set (R1.8).

Testing eight metrics at alpha 0.05 and reporting the one that moved gives roughly a
1 in 3 chance of a false positive somewhere. Correction is what makes a multi-metric
readout honest.

**The precondition matters more than the method.** A correction is only meaningful if
the metric set was fixed *in advance*: dividing by ``m`` is worthless if ``m`` grows as
the analyst keeps looking. That is what ``ExperimentSpec.all_metrics`` is for, and why
:func:`correct_spec_metrics` takes a spec rather than a bare list.

**Which correction.** Bonferroni controls the family-wise error rate -- the chance of
*any* false positive -- and is conservative. Benjamini-Hochberg controls the false
discovery rate -- the expected *proportion* of rejections that are false -- and is more
powerful. For a readout with one primary metric and a handful of guardrails, BH is the
better default: a guardrail false positive costs an investigation, not a wrong ship
decision, so trading some FWER control for power is the right call. Bonferroni is
available for a genuinely confirmatory family where any false positive is unacceptable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

import numpy as np

__all__ = ["CorrectionResult", "Method", "correct", "correct_spec_metrics"]

Method = Literal["benjamini_hochberg", "bonferroni", "none"]


class CorrectionResult(NamedTuple):
    """Adjusted p-values and rejection decisions, in the input order."""

    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]
    method: Method
    alpha: float
    n_tests: int

    @property
    def n_rejected(self) -> int:
        return sum(self.rejected)


def correct(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
    method: Method = "benjamini_hochberg",
) -> CorrectionResult:
    """Adjust ``p_values`` for multiplicity.

    Parameters
    ----------
    p_values
        Raw p-values, one per test in the pre-declared family.
    alpha
        Family-wise error rate (Bonferroni) or false discovery rate (BH).
    method
        ``"benjamini_hochberg"`` (default), ``"bonferroni"``, or ``"none"``.

    Returns
    -------
    CorrectionResult
        ``adjusted`` and ``rejected`` are in the same order as the input.

    Notes
    -----
    BH adjusted p-values are the *step-up* values, enforced monotone::

        p_adj[i] = min over j >= i of ( m / rank_j * p_j )    (sorted ascending)

    The cumulative minimum from the largest p-value downward is what guarantees
    monotonicity: without it an adjusted p-value could come out smaller than one for a
    more significant test. Matches ``statsmodels.stats.multitest.multipletests``, which
    the reference tests assert.

    ``method="none"`` returns the raw p-values unchanged. It exists so a caller can
    make "no correction" an explicit, visible choice rather than an omission.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if len(p_values) == 0:
        raise ValueError("no p-values supplied; nothing to correct")

    p = np.asarray(p_values, dtype=float)
    if np.any((p < 0.0) | (p > 1.0)) or np.any(np.isnan(p)):
        raise ValueError(f"p-values must all be in [0, 1] and non-NaN, got {list(p_values)}")

    m = p.size

    if method == "none":
        return CorrectionResult(
            adjusted=tuple(p.tolist()),
            rejected=tuple((p <= alpha).tolist()),
            method="none",
            alpha=alpha,
            n_tests=m,
        )

    if method == "bonferroni":
        adjusted = np.minimum(p * m, 1.0)
        return CorrectionResult(
            adjusted=tuple(adjusted.tolist()),
            rejected=tuple((adjusted <= alpha).tolist()),
            method="bonferroni",
            alpha=alpha,
            n_tests=m,
        )

    if method != "benjamini_hochberg":
        raise ValueError(f"unknown correction method {method!r}")

    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    raw = p_sorted * m / ranks
    # Step-up: sweep from the largest p-value down so the result is monotone.
    adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return CorrectionResult(
        adjusted=tuple(adjusted.tolist()),
        rejected=tuple((adjusted <= alpha).tolist()),
        method="benjamini_hochberg",
        alpha=alpha,
        n_tests=m,
    )


def correct_spec_metrics(
    p_by_metric: Mapping[str, float],
    spec_metrics: Sequence[str],
    *,
    alpha: float = 0.05,
    method: Method = "benjamini_hochberg",
) -> dict[str, tuple[float, bool]]:
    """Correct across the **pre-declared** metric family, keyed by metric name.

    Enforces the precondition that makes correction meaningful: every metric tested
    must be in the spec's declared set, and every declared metric must have been
    tested. A family that silently grows or shrinks makes the divisor meaningless
    (R1.8).

    Parameters
    ----------
    p_by_metric
        Raw p-value per metric.
    spec_metrics
        ``ExperimentSpec.all_metrics`` -- the family declared before analysis.
    alpha, method
        As :func:`correct`.

    Returns
    -------
    dict[str, tuple[float, bool]]
        ``{metric: (adjusted_p, rejected)}``.

    Raises
    ------
    ValueError
        If the tested metrics do not exactly match the declared family.
    """
    tested, declared = set(p_by_metric), set(spec_metrics)
    if extra := tested - declared:
        raise ValueError(
            f"metric(s) {sorted(extra)} were tested but are not in the declared family "
            f"{sorted(declared)}. Correcting over an expanded family understates the "
            "multiplicity; add them to a NEW spec or label the analysis exploratory (R1.8)."
        )
    if missing := declared - tested:
        raise ValueError(
            f"declared metric(s) {sorted(missing)} have no p-value. Dropping a declared "
            "metric shrinks the divisor and overstates significance; test the whole "
            "family or amend the spec (R1.8)."
        )

    names = list(spec_metrics)
    result = correct([p_by_metric[n] for n in names], alpha=alpha, method=method)
    return {n: (result.adjusted[i], result.rejected[i]) for i, n in enumerate(names)}
