"""Sample ratio mismatch.

An SRM means the observed split between arms differs from the intended split by more
than chance. It is the single highest-value sanity check in online
experimentation, because it usually indicates that *assignment or logging is
broken* -- bot filtering that hits one arm harder, a redirect that drops users, a
telemetry bug. When that is true, no amount of statistical sophistication downstream
rescues the result (R1.3).

The default threshold is ``p < 0.0005``, following Kohavi et al. It is deliberately
far stricter than 0.05: this test runs on every experiment, so at 0.05 roughly one
healthy experiment in twenty would be flagged, and a check that cries wolf gets
ignored. The cost asymmetry runs the other way too -- a missed SRM invalidates a
shipped decision.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import stats

from gatekeeper.data.schema import ExperimentData
from gatekeeper.types import InsufficientData, SanityCheck

__all__ = ["DEFAULT_SRM_THRESHOLD", "check_srm", "srm_test"]

DEFAULT_SRM_THRESHOLD = 0.0005
"""Kohavi's recommended p-value threshold for flagging an SRM."""


def srm_test(
    observed: Mapping[str, int],
    expected_shares: Mapping[str, float] | None = None,
) -> tuple[float, float]:
    """Chi-square goodness-of-fit test of observed arm counts against intended shares.

    Parameters
    ----------
    observed
        Unit count per arm.
    expected_shares
        Intended share per arm; must sum to 1. Defaults to an equal split across the
        arms present in ``observed``.

    Returns
    -------
    tuple[float, float]
        ``(chi2_statistic, p_value)`` with ``len(observed) - 1`` degrees of freedom.

    Raises
    ------
    InsufficientData
        If fewer than two arms are present, or the total count is zero.
    ValueError
        If shares do not sum to 1, are non-positive, or do not match the arms in
        ``observed``.

    Examples
    --------
    A perfectly balanced split has a chi-square statistic of exactly zero:

    >>> chi2, p = srm_test({"a": 500, "b": 500})
    >>> round(chi2, 10), round(p, 6)
    (0.0, 1.0)

    A 600/400 split on an intended 50/50 gives ``(600-500)^2/500 * 2 = 40``:

    >>> chi2, p = srm_test({"a": 600, "b": 400})
    >>> round(chi2, 10)
    40.0
    """
    if len(observed) < 2:
        raise InsufficientData(
            f"SRM needs at least two arms, got {len(observed)}: {sorted(observed)}"
        )

    arms = sorted(observed)
    counts = np.array([observed[a] for a in arms], dtype=float)
    if np.any(counts < 0):
        raise ValueError(f"arm counts must be non-negative, got {dict(observed)}")

    total = counts.sum()
    if total == 0:
        raise InsufficientData("SRM needs at least one unit; all arm counts are zero")

    if expected_shares is None:
        shares = np.full(len(arms), 1.0 / len(arms))
    else:
        if set(expected_shares) != set(arms):
            raise ValueError(
                f"expected_shares keys {sorted(expected_shares)} do not match observed arms {arms}"
            )
        shares = np.array([expected_shares[a] for a in arms], dtype=float)
        if np.any(shares <= 0):
            raise ValueError(f"expected shares must be positive, got {dict(expected_shares)}")
        if not np.isclose(shares.sum(), 1.0):
            raise ValueError(f"expected shares must sum to 1, got {shares.sum():.6f}")

    expected = shares * total
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    p_value = float(stats.chi2.sf(chi2, df=len(arms) - 1))
    return chi2, p_value


def check_srm(
    data: ExperimentData,
    expected_shares: Mapping[str, float] | None = None,
    *,
    threshold: float = DEFAULT_SRM_THRESHOLD,
) -> SanityCheck:
    """Run the SRM check and package it for the sanity gate.

    Parameters
    ----------
    data
        The experiment to check.
    expected_shares
        Intended split. Defaults to equal shares across the arms present.
    threshold
        Flag an SRM when ``p < threshold``. Comes from the spec in normal use, so
        the threshold is fixed *before* the observed split is looked at (R1.2).

    Returns
    -------
    SanityCheck
        ``passed=False`` when ``p < threshold``.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    # The arm universe must come from what was *intended*, not from what arrived.
    # An arm that vanished entirely is the most extreme SRM there is, and reading the
    # universe off the data would hide it: n_per_arm reports only arms present, so a
    # missing arm would silently reduce this to a one-arm test and raise instead of
    # reporting a failure. The gate must block, not crash (R1.3).
    universe = tuple(expected_shares) if expected_shares is not None else data.schema.variants
    present = data.n_per_arm
    unknown = set(present) - set(universe)
    if unknown:
        raise ValueError(
            f"data contains arm(s) {sorted(unknown)} absent from the intended allocation "
            f"{sorted(universe)}; the spec does not describe this experiment"
        )
    observed = {arm: present.get(arm, 0) for arm in universe}
    chi2, p_value = srm_test(observed, expected_shares)

    total = sum(observed.values())
    obs_desc = "  ".join(f"{k}={v:,} ({v / total:.2%})" for k, v in sorted(observed.items()))
    passed = p_value >= threshold

    if passed:
        detail = (
            f"Split consistent with intended allocation. {obs_desc}  "
            f"chi2={chi2:.3f}  p={p_value:.4g} (>= {threshold:g})"
        )
    else:
        vanished = sorted(arm for arm, n in observed.items() if n == 0)
        prefix = (
            f"SAMPLE RATIO MISMATCH -- arm(s) {vanished} received ZERO units. "
            if vanished
            else "SAMPLE RATIO MISMATCH: "
        )
        detail = (
            f"{prefix}Observed split is inconsistent with the intended allocation. "
            f"{obs_desc}  chi2={chi2:.3f}  p={p_value:.4g} (< {threshold:g}). "
            "Assignment or logging is likely broken; metric results are not "
            "trustworthy until this is explained."
        )

    return SanityCheck(
        name="srm",
        passed=passed,
        detail=detail,
        statistic=chi2,
        p_value=p_value,
        threshold=threshold,
    )
