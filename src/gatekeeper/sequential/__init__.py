"""Sequential testing: monitoring an experiment without inflating the error rate.

Always-valid p-values (mSPRT) are implemented; alpha-spending boundaries
(O'Brien-Fleming, Pocock) are deferred -- see
:mod:`gatekeeper.sequential.always_valid` for the reasoning.
"""

from __future__ import annotations

from gatekeeper.sequential.always_valid import (
    SequentialResult,
    always_valid_interval,
    always_valid_p_value,
    sequential_p_values,
    suggest_tau,
)
from gatekeeper.sequential.peeking import PeekingResult, simulate_peeking

__all__ = [
    "PeekingResult",
    "SequentialResult",
    "always_valid_interval",
    "always_valid_p_value",
    "sequential_p_values",
    "simulate_peeking",
    "suggest_tau",
]
