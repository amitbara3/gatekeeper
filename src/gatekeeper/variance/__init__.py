"""Variance-reduction techniques.

Currently CUPED. Note that Cookie Cats has no pre-experiment covariate, so nothing here
can be applied to it -- see :mod:`gatekeeper.variance.cuped` for why using
``sum_gamerounds`` instead would be bias dressed as a win (R1.7, PRD §6).
"""

from __future__ import annotations

from gatekeeper.variance.cuped import (
    CupedAdjustment,
    cuped_adjust,
    cuped_theta,
    estimate_cuped,
)

__all__ = ["CupedAdjustment", "cuped_adjust", "cuped_theta", "estimate_cuped"]
