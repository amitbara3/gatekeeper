"""Assignment-integrity checks.

These answer "is this experiment trustworthy?" rather than "can this file be
parsed?" -- the latter is :mod:`gatekeeper.data.schema`. Each check returns a
:class:`~gatekeeper.types.SanityCheck` so failures surface as a blockable readout
state (Design §4.2) instead of a crash, and so an override is recorded rather than
silent (R1.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from gatekeeper.data.schema import ExperimentData
from gatekeeper.design.srm import DEFAULT_SRM_THRESHOLD, check_srm
from gatekeeper.types import SanityCheck, SanityReport

__all__ = [
    "check_arm_sizes",
    "check_no_cross_arm_units",
    "check_unique_units",
    "run_sanity_checks",
]


def check_unique_units(data: ExperimentData) -> SanityCheck:
    """Verify the randomisation unit appears at most once.

    Duplicated units break the independence assumption behind every standard
    variance formula, which shrinks standard errors and manufactures significance
    (R1.13).
    """
    unit_col = data.schema.unit_col
    dupes = data.frame[unit_col].duplicated()
    n_dupes = int(dupes.sum())
    n_affected = int(
        data.frame.loc[data.frame[unit_col].duplicated(keep=False), unit_col].nunique()
    )

    if n_dupes == 0:
        return SanityCheck(
            name="unique_units",
            passed=True,
            detail=f"All {len(data.frame):,} rows have a distinct {unit_col}.",
            statistic=0.0,
        )

    examples = data.frame.loc[dupes, unit_col].head(5).tolist() if n_dupes else []
    return SanityCheck(
        name="unique_units",
        passed=False,
        detail=(
            f"{n_dupes:,} duplicate row(s) across {n_affected:,} distinct {unit_col} value(s) "
            f"(e.g. {examples}). Duplicated units violate independence and deflate "
            "standard errors. Deduplicate upstream and explain the cause."
        ),
        statistic=float(n_dupes),
    )


def check_no_cross_arm_units(data: ExperimentData) -> SanityCheck:
    """Verify no unit appears in more than one arm.

    A unit in both arms means assignment is not sticky -- a serious instrumentation
    bug that contaminates both arms and biases the estimate toward zero.
    """
    unit_col, variant_col = data.schema.unit_col, data.schema.variant_col
    arms_per_unit = data.frame.groupby(unit_col, observed=True)[variant_col].nunique()
    offenders = arms_per_unit[arms_per_unit > 1]
    n_offenders = len(offenders)

    if n_offenders == 0:
        return SanityCheck(
            name="no_cross_arm_units",
            passed=True,
            detail=f"No {unit_col} appears in more than one arm.",
            statistic=0.0,
        )

    return SanityCheck(
        name="no_cross_arm_units",
        passed=False,
        detail=(
            f"{n_offenders:,} {unit_col} value(s) appear in more than one arm "
            f"(e.g. {offenders.index[:5].tolist()}). Assignment is not sticky; both arms "
            "are contaminated and the effect is biased toward zero."
        ),
        statistic=float(n_offenders),
    )


def check_arm_sizes(data: ExperimentData, *, min_per_arm: int = 100) -> SanityCheck:
    """Verify every declared arm is present with a usable number of units."""
    if min_per_arm < 1:
        raise ValueError(f"min_per_arm must be at least 1, got {min_per_arm}")

    counts = data.n_per_arm
    declared = set(data.schema.variants)
    present = set(counts)
    missing = declared - present
    undersized = {k: v for k, v in counts.items() if v < min_per_arm}

    if not missing and not undersized:
        arms = "  ".join(f"{k}={v:,}" for k, v in sorted(counts.items()))
        return SanityCheck(
            name="arm_sizes",
            passed=True,
            detail=f"All arms present with >= {min_per_arm:,} units. {arms}",
            statistic=float(min(counts.values())),
            threshold=float(min_per_arm),
        )

    problems = []
    if missing:
        problems.append(f"arm(s) absent from the data: {sorted(missing)}")
    if undersized:
        problems.append(f"arm(s) below {min_per_arm:,} units: {undersized}")
    return SanityCheck(
        name="arm_sizes",
        passed=False,
        detail="; ".join(problems) + ". Estimates on these arms are not meaningful.",
        statistic=float(min(counts.values())) if counts else 0.0,
        threshold=float(min_per_arm),
    )


def run_sanity_checks(
    data: ExperimentData,
    *,
    expected_shares: Mapping[str, float] | None = None,
    srm_threshold: float = DEFAULT_SRM_THRESHOLD,
    min_per_arm: int = 100,
    extra: Sequence[SanityCheck] = (),
) -> SanityReport:
    """Run the standard gate: SRM, uniqueness, cross-arm leakage, arm sizes.

    This is what ``analyze()`` consumes. Every check runs even if an earlier one
    fails, so the report shows the full picture rather than only the first problem.

    Parameters
    ----------
    data
        The experiment to check.
    expected_shares
        Intended allocation for the SRM test. Defaults to an equal split.
    srm_threshold
        SRM p-value threshold; should come from the spec (R1.2).
    min_per_arm
        Minimum units per arm.
    extra
        Additional pre-computed checks to fold into the report.

    Returns
    -------
    SanityReport
        Use :meth:`~gatekeeper.types.SanityReport.raise_if_failed` to gate analysis.
    """
    checks = [
        check_srm(data, expected_shares, threshold=srm_threshold),
        check_unique_units(data),
        check_no_cross_arm_units(data),
        check_arm_sizes(data, min_per_arm=min_per_arm),
        *extra,
    ]
    return SanityReport(checks=tuple(checks))
