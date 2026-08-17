"""Uplift and Qini curves: is a CATE ranking actually useful?

A CATE model produces a per-unit score. The question that matters operationally is not
"is the model accurate" but "if I target the top k% by this score, how much more effect do
I capture than by targeting at random?" Uplift and Qini curves answer exactly that.

**Uplift curve.** Sort units by predicted effect, descending. At each depth, compute the
cumulative gain::

    gain(k) = ( sum_treated_y(k) / n_treated(k) - sum_control_y(k) / n_control(k) ) * n(k)

That is the difference in mean outcome among the top k, scaled back up by how many units
are involved -- the total effect captured by targeting that group.

**Qini curve** is the variant that scales by the treated count instead, making it robust
to imbalanced targeting. In practice the two are close; Qini is the one usually quoted.

**The area is the summary.** Area between the curve and the random-targeting diagonal
divided by the area a perfect ranking would achieve gives a normalised Qini coefficient:
1.0 is perfect ranking, 0 is no better than random, and **negative is worse than random**
-- which happens, and is a real result rather than a bug. A CATE model that ranks
backwards is worse than no model, because it would have you target exactly the people the
treatment harms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gatekeeper.types import InsufficientData

__all__ = ["UpliftCurve", "qini_curve", "uplift_curve"]


@dataclass(frozen=True, slots=True)
class UpliftCurve:
    """A cumulative-gain curve for a CATE ranking."""

    depths: np.ndarray
    """Fraction of the population targeted, ascending from >0 to 1."""
    gains: np.ndarray
    """Cumulative effect captured at each depth."""
    kind: str
    """``"uplift"`` or ``"qini"``."""

    @property
    def total_gain(self) -> float:
        """Gain at full depth -- equals the overall effect times n."""
        return float(self.gains[-1])

    @property
    def area(self) -> float:
        """Area under the curve, by the trapezoidal rule."""
        return float(np.trapezoid(self.gains, self.depths))

    @property
    def random_area(self) -> float:
        """Area a random ranking would achieve -- the straight diagonal to total gain."""
        return float(self.total_gain / 2.0)

    @property
    def coefficient(self) -> float:
        """Normalised Qini/uplift coefficient.

        ``(area - random_area) / |random_area|``. Zero means the ranking is no better than
        random; **negative means worse than random**, which is a genuine finding: a model
        ranking backwards would have you target the people the treatment hurts.
        """
        if self.random_area == 0:
            return 0.0
        return float((self.area - self.random_area) / abs(self.random_area))

    @property
    def beats_random(self) -> bool:
        return self.coefficient > 0.0

    def describe(self) -> str:
        verdict = "better than random" if self.beats_random else "NO BETTER than random targeting"
        return (
            f"{self.kind} curve: coefficient {self.coefficient:+.4f} ({verdict}), "
            f"total gain {self.total_gain:+.2f} over {self.depths.size} depths"
        )


def _curve(
    scores: np.ndarray,
    treated: np.ndarray,
    outcome: np.ndarray,
    *,
    n_bins: int,
    kind: str,
) -> UpliftCurve:
    if not (scores.size == treated.size == outcome.size):
        raise ValueError(
            f"scores, treated, and outcome must align: got {scores.size}, "
            f"{treated.size}, {outcome.size}"
        )
    n = scores.size
    if n < n_bins * 2:
        raise InsufficientData(f"need at least {n_bins * 2} units for {n_bins} bins, got {n}")
    if not set(np.unique(treated)) <= {0.0, 1.0}:
        raise ValueError("treated must be a 0/1 indicator")

    # Descending by predicted effect: target the most responsive first.
    order = np.argsort(-scores, kind="stable")
    t_sorted = treated[order]
    y_sorted = outcome[order]

    cut_points = np.linspace(n / n_bins, n, n_bins).astype(int)
    depths: list[float] = []
    gains: list[float] = []
    for k in cut_points:
        t_head = t_sorted[:k]
        y_head = y_sorted[:k]
        n_treated = float(t_head.sum())
        n_control = float(k - n_treated)
        if n_treated == 0 or n_control == 0:
            # Cannot form a contrast at this depth; carry the previous gain forward
            # rather than inventing one.
            gains.append(gains[-1] if gains else 0.0)
            depths.append(k / n)
            continue

        mean_t = float(y_head[t_head == 1.0].mean())
        mean_c = float(y_head[t_head == 0.0].mean())
        scale = n_treated if kind == "qini" else float(k)
        gains.append((mean_t - mean_c) * scale)
        depths.append(k / n)

    return UpliftCurve(
        depths=np.array(depths, dtype=float),
        gains=np.array(gains, dtype=float),
        kind=kind,
    )


def uplift_curve(
    scores: np.ndarray,
    treated: np.ndarray,
    outcome: np.ndarray,
    *,
    n_bins: int = 20,
) -> UpliftCurve:
    """Uplift curve, scaling each depth's contrast by the total units targeted."""
    return _curve(scores, treated, outcome, n_bins=n_bins, kind="uplift")


def qini_curve(
    scores: np.ndarray,
    treated: np.ndarray,
    outcome: np.ndarray,
    *,
    n_bins: int = 20,
) -> UpliftCurve:
    """Qini curve, scaling each depth's contrast by the treated count.

    Preferred over the plain uplift curve when targeting is imbalanced across depths,
    since scaling by the treated count keeps the comparison stable.
    """
    return _curve(scores, treated, outcome, n_bins=n_bins, kind="qini")
