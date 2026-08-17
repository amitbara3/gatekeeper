"""Sanity checks -- the gate between data and analysis (Architecture §2.1)."""

from __future__ import annotations

from gatekeeper.checks.integrity import (
    check_arm_sizes,
    check_no_cross_arm_units,
    check_unique_units,
    run_sanity_checks,
)
from gatekeeper.checks.outliers import OutlierProfile, check_outlier_leverage, profile_metric

__all__ = [
    "OutlierProfile",
    "check_arm_sizes",
    "check_no_cross_arm_units",
    "check_outlier_leverage",
    "check_unique_units",
    "profile_metric",
    "run_sanity_checks",
]
