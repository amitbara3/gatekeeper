"""Design.md's palette, as code. The only place colour hex values appear (R4.8).

Every value here was validated with the six-check colourblindness/contrast validator in
**both** light and dark mode, using the harder all-pairs test because propensity-overlap
and CUPED scatter plots put any two marks side by side. Results are recorded in Design.md
§2; the summary is that all checks pass, with one documented WARN.

**The WARN matters and is handled here.** Light-mode aqua (slot 3) sits at 2.74:1 against
the light surface, below the 3:1 threshold for marks. Design.md's relief rule therefore
applies: wherever slot 3 appears on the light surface, the chart must ship visible direct
labels or a table view. :func:`requires_relief` exists so a plotting function can check
that mechanically rather than relying on someone remembering.

**The other collision worth knowing.** Treatment-orange sits only ΔE ≈ 5.8 from
status-serious in light mode. A bare orange "HOLD" chip beside an orange treatment series
is genuinely ambiguous, so status colours always ship with an icon and a word, and never
inside the plot area. :data:`STATUS_ICONS` is why.

Three colour jobs are kept strictly separate, because conflating them is the most common
charting error:

- **variant identity** is categorical -- fixed slot order, never cycled, never reassigned
  by rank, so a filter that changes the series count cannot repaint the survivors;
- **magnitude** is sequential -- one hue, light to dark;
- **decision state** is status -- a reserved scale that never doubles as "series 4".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gatekeeper.types import Decision

__all__ = [
    "MAX_CATEGORICAL_SERIES",
    "STATUS_ICONS",
    "Mode",
    "Palette",
    "decision_style",
    "palette",
    "requires_relief",
    "variant_colors",
]

Mode = Literal["light", "dark"]

MAX_CATEGORICAL_SERIES = 3
"""Hard cap on categorical series.

Not a stylistic preference: only the first three slots clear the all-pairs
colourblindness floors in both modes. A fourth would put yellow beside orange, which
fails. Past three, fold into "Other" or facet into small multiples -- the answer is fewer
series, never a wider palette.
"""

STATUS_ICONS: dict[Decision, str] = {
    Decision.SHIP: "✔",
    Decision.HOLD: "▲",
    Decision.INCONCLUSIVE: "●",
    Decision.BLOCKED: "✕",
}
"""Status colour is never allowed to carry meaning alone.

On the light surface ``warning`` (1.79:1) and ``serious`` (2.57:1) are sub-3:1 by design,
and treatment-orange sits close to status-serious. The icon plus a word is the mitigation.
"""


@dataclass(frozen=True, slots=True)
class Palette:
    """Every colour role for one mode."""

    mode: Mode

    # Categorical -- variant identity, fixed order.
    series: tuple[str, str, str]
    # Sequential -- magnitude, light to dark.
    sequential: tuple[str, str, str, str, str]
    # Diverging -- polarity, with a NEUTRAL GRAY midpoint (never a hue at zero).
    diverging_negative: str
    diverging_neutral: str
    diverging_positive: str
    # Status -- reserved, never themed, never a series colour.
    status_good: str
    status_warning: str
    status_serious: str
    status_critical: str
    # Chrome and ink.
    surface: str
    page: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    gridline: str
    baseline: str
    delta_positive: str

    @property
    def control(self) -> str:
        """Slot 1 -- always the control arm."""
        return self.series[0]

    @property
    def treatment(self) -> str:
        """Slot 2 -- always the treatment arm."""
        return self.series[1]

    def series_color(self, index: int) -> str:
        """Colour for series ``index`` (0-based), refusing to cycle.

        Cycling would silently give two series the same colour. Raising forces the caller
        to fold into "Other" or facet, which is the correct fix.
        """
        if not 0 <= index < MAX_CATEGORICAL_SERIES:
            raise ValueError(
                f"series index {index} is outside the validated palette "
                f"(0..{MAX_CATEGORICAL_SERIES - 1}). Only the first "
                f"{MAX_CATEGORICAL_SERIES} slots clear the all-pairs colourblindness "
                "floors, so a fourth series must fold into 'Other' or become a small "
                "multiple -- the palette is not extended."
            )
        return self.series[index]


_LIGHT = Palette(
    mode="light",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    sequential=("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"),
    diverging_negative="#d03b3b",
    diverging_neutral="#f0efec",
    diverging_positive="#2a78d6",
    status_good="#0ca30c",
    status_warning="#fab219",
    status_serious="#ec835a",
    status_critical="#d03b3b",
    surface="#fcfcfb",
    page="#f9f9f7",
    ink_primary="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    gridline="#e1e0d9",
    baseline="#c3c2b7",
    delta_positive="#006300",
)

_DARK = Palette(
    mode="dark",
    # The same three hues, re-stepped for the dark surface -- NOT an inverted light
    # palette. Each was validated against #1a1a19 separately.
    series=("#3987e5", "#d95926", "#199e70"),
    sequential=("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#184f95"),
    diverging_negative="#d03b3b",
    diverging_neutral="#383835",
    diverging_positive="#3987e5",
    # Status is fixed and never themed: all four clear 3:1 on the dark surface too.
    status_good="#0ca30c",
    status_warning="#fab219",
    status_serious="#ec835a",
    status_critical="#d03b3b",
    surface="#1a1a19",
    page="#0d0d0d",
    ink_primary="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    gridline="#2c2c2a",
    baseline="#383835",
    delta_positive="#0ca30c",
)


def palette(mode: Mode = "light") -> Palette:
    """The palette for ``mode``."""
    if mode == "light":
        return _LIGHT
    if mode == "dark":
        return _DARK
    raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")


def variant_colors(variants: tuple[str, ...], mode: Mode = "light") -> dict[str, str]:
    """Map variant labels to colours by **position, not by rank**.

    Colour follows the entity. Sorting a chart or filtering out a variant must never
    repaint the survivors, which is what would happen if colours were assigned by value
    or by iteration order of a set.
    """
    p = palette(mode)
    if len(variants) > MAX_CATEGORICAL_SERIES:
        raise ValueError(
            f"{len(variants)} variants exceeds the validated palette's "
            f"{MAX_CATEGORICAL_SERIES} slots: {list(variants)}. Fold the extras into "
            "'Other' or use small multiples."
        )
    return {name: p.series_color(i) for i, name in enumerate(variants)}


def requires_relief(color: str, mode: Mode = "light") -> bool:
    """Whether this colour needs visible labels or a table view to be legible.

    True for light-mode slot 3 (aqua, 2.74:1 against the light surface). Design.md's
    contrast WARN is **not dismissable** -- it obligates a relief channel -- so exposing
    it as a function lets a plotting helper enforce that instead of trusting memory.
    """
    return mode == "light" and color == _LIGHT.series[2]


def decision_style(decision: Decision, mode: Mode = "light") -> tuple[str, str]:
    """``(colour, icon)`` for a decision state.

    Always used as a pair. The icon is not decoration: it is what stops the status colour
    from carrying meaning alone, which matters both for sub-3:1 light-mode status colours
    and for the orange/serious collision described in the module docstring.
    """
    p = palette(mode)
    colors = {
        Decision.SHIP: p.status_good,
        Decision.HOLD: p.status_serious,
        Decision.INCONCLUSIVE: p.status_warning,
        Decision.BLOCKED: p.status_critical,
    }
    return colors[decision], STATUS_ICONS[decision]
