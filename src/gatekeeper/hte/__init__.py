"""Heterogeneous treatment effects: who is affected, and is the ranking useful?

Cookie Cats has no pre-treatment covariates, so honest CATE estimation on it is
impossible. These modules are validated against synthetic CATE functions known by
construction, and saying so is Phase 7's correct outcome rather than a shortfall
(PRD §6).
"""

from __future__ import annotations

from gatekeeper.hte.learners import CateEstimate, LearnerKind, estimate_cate
from gatekeeper.hte.uplift import UpliftCurve, qini_curve, uplift_curve

__all__ = [
    "CateEstimate",
    "LearnerKind",
    "UpliftCurve",
    "estimate_cate",
    "qini_curve",
    "uplift_curve",
]
