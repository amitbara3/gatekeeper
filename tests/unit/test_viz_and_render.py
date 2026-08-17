"""The palette (R4.8) and the HTML/Markdown renderers."""

from __future__ import annotations

import re

import pytest

from gatekeeper.report.readout import build_readout
from gatekeeper.report.render import render_html, render_markdown
from gatekeeper.spec import ExperimentSpec
from gatekeeper.types import (
    DataSource,
    Decision,
    EffectEstimate,
    Estimand,
    SanityCheck,
    SanityReport,
)
from gatekeeper.viz.theme import (
    MAX_CATEGORICAL_SERIES,
    STATUS_ICONS,
    decision_style,
    palette,
    requires_relief,
    variant_colors,
)

PASSING = SanityReport(checks=(SanityCheck("srm", True, "split is fine"),))
FAILING = SanityReport(checks=(SanityCheck("srm", False, "SAMPLE RATIO MISMATCH"),))


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="test",
        dataset="cookie_cats",
        registered_on="2026-08-17",
        primary_metric="retention_7",
        direction="higher_is_better",
        guardrail_metrics=("retention_1",),
        mde=0.0075,
        practical_threshold=0.01,
        expected_shares={"gate_30": 0.5, "gate_40": 0.5},
    )


def estimate(metric: str, point: float, ci: tuple[float, float], p: float) -> EffectEstimate:
    return EffectEstimate(
        estimand=Estimand(outcome=metric, treatment="version"),
        point=point,
        ci=ci,
        method="two_proportion_z",
        assumptions=("independent units",),
        data_source=DataSource.SYNTHETIC,
        n_per_arm={"gate_30": 45_000, "gate_40": 45_000},
        p_value=p,
    )


def shipping_readout():
    return build_readout(
        spec(),
        PASSING,
        {
            "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
            "retention_1": estimate("retention_1", 0.001, (-0.004, 0.006), 0.7),
        },
    )


def blocked_readout():
    return build_readout(
        spec(),
        FAILING,
        {
            "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
            "retention_1": estimate("retention_1", 0.001, (-0.004, 0.006), 0.7),
        },
    )


class TestPalette:
    def test_both_modes_are_available(self):
        assert palette("light").mode == "light"
        assert palette("dark").mode == "dark"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            palette("sepia")  # type: ignore[arg-type]

    def test_control_is_slot_one_and_treatment_slot_two(self):
        p = palette("light")
        assert p.control == p.series[0]
        assert p.treatment == p.series[1]

    def test_dark_is_a_reselected_palette_not_an_inversion(self):
        """The dark hues are re-stepped for the dark surface, not flipped."""
        light, dark = palette("light"), palette("dark")
        assert light.series != dark.series
        # Same hue families though: blue, orange, aqua in the same order.
        assert len(light.series) == len(dark.series) == MAX_CATEGORICAL_SERIES

    def test_status_colors_are_not_themed(self):
        """Status is a reserved fixed scale; it must not change between modes."""
        light, dark = palette("light"), palette("dark")
        for role in ("status_good", "status_warning", "status_serious", "status_critical"):
            assert getattr(light, role) == getattr(dark, role), role

    def test_status_colors_are_never_series_colors(self):
        for mode in ("light", "dark"):
            p = palette(mode)  # type: ignore[arg-type]
            statuses = {
                p.status_good,
                p.status_warning,
                p.status_serious,
                p.status_critical,
            }
            assert not statuses & set(p.series), f"{mode}: status colour reused as a series"

    def test_diverging_midpoint_is_neutral_not_a_hue(self):
        """A hue at zero would imply direction where there is none."""
        for mode in ("light", "dark"):
            p = palette(mode)  # type: ignore[arg-type]
            r, g, b = (int(p.diverging_neutral[i : i + 2], 16) for i in (1, 3, 5))
            assert max(r, g, b) - min(r, g, b) < 20, f"{mode} midpoint is not neutral"

    def test_sequential_ramp_is_monotone_in_lightness(self):
        for mode in ("light", "dark"):
            p = palette(mode)  # type: ignore[arg-type]
            luminance = [sum(int(step[i : i + 2], 16) for i in (1, 3, 5)) for step in p.sequential]
            assert luminance == sorted(luminance, reverse=True), f"{mode} ramp not monotone"

    def test_every_color_is_a_valid_hex(self):
        for mode in ("light", "dark"):
            p = palette(mode)  # type: ignore[arg-type]
            for value in (*p.series, *p.sequential, p.surface, p.ink_primary):
                assert re.fullmatch(r"#[0-9a-f]{6}", value), value


class TestSeriesCap:
    def test_refuses_to_cycle(self):
        """Cycling would silently give two series the same colour."""
        p = palette("light")
        with pytest.raises(ValueError, match="outside the validated palette"):
            p.series_color(MAX_CATEGORICAL_SERIES)

    def test_error_explains_the_correct_fix(self):
        p = palette("light")
        with pytest.raises(ValueError, match=r"'Other'|small multiple"):
            p.series_color(7)

    def test_negative_index_refused(self):
        with pytest.raises(ValueError):
            palette("light").series_color(-1)

    def test_variant_colors_assign_by_position(self):
        colors = variant_colors(("gate_30", "gate_40"), "light")
        p = palette("light")
        assert colors["gate_30"] == p.series[0]
        assert colors["gate_40"] == p.series[1]

    def test_variant_colors_are_stable_under_reordering_of_the_data(self):
        """Colour follows the entity; a re-sorted chart must not repaint."""
        first = variant_colors(("gate_30", "gate_40"), "light")
        again = variant_colors(("gate_30", "gate_40"), "light")
        assert first == again

    def test_too_many_variants_refused(self):
        with pytest.raises(ValueError, match="exceeds the validated palette"):
            variant_colors(("a", "b", "c", "d"), "light")


class TestReliefRule:
    def test_light_slot_three_needs_relief(self):
        """Aqua sits at 2.74:1 on the light surface -- a non-dismissable WARN."""
        assert requires_relief(palette("light").series[2], "light")

    def test_slots_one_and_two_do_not(self):
        p = palette("light")
        assert not requires_relief(p.series[0], "light")
        assert not requires_relief(p.series[1], "light")

    def test_dark_mode_needs_no_relief(self):
        assert not requires_relief(palette("dark").series[2], "dark")


class TestDecisionStyle:
    def test_every_decision_has_a_colour_and_an_icon(self):
        for decision in Decision:
            color, icon = decision_style(decision)
            assert re.fullmatch(r"#[0-9a-f]{6}", color)
            assert icon

    def test_icons_are_distinct(self):
        assert len(set(STATUS_ICONS.values())) == len(Decision)

    def test_ship_and_blocked_differ(self):
        assert decision_style(Decision.SHIP)[0] != decision_style(Decision.BLOCKED)[0]

    def test_icon_is_always_paired_with_the_colour(self):
        """Status colour must never carry meaning alone: orange sits close to
        status-serious in light mode, and two status colours are sub-3:1."""
        for decision in Decision:
            _, icon = decision_style(decision, "light")
            assert icon == STATUS_ICONS[decision]


class TestRenderHtml:
    def test_is_self_contained(self):
        html = render_html(shipping_readout())
        for forbidden in ("<script", "http://", "https://", "@import", "<link"):
            assert forbidden not in html, f"external resource: {forbidden}"

    def test_omits_document_scaffolding(self):
        """The Artifact wrapper supplies doctype/html/head/body."""
        html = render_html(shipping_readout())
        for tag in ("<!doctype", "<html", "<head>", "<body"):
            assert tag not in html.lower()

    def test_includes_a_title(self):
        assert "<title>" in render_html(shipping_readout())

    def test_defines_both_light_and_dark_palettes(self):
        html = render_html(shipping_readout())
        assert "prefers-color-scheme: dark" in html
        assert '[data-theme="dark"]' in html
        assert palette("light").surface in html
        assert palette("dark").surface in html

    def test_body_paints_its_own_background(self):
        """A transparent body would borrow the host page's colours."""
        html = render_html(shipping_readout())
        assert re.search(r"body\s*\{[^}]*background:", html)

    def test_shows_the_decision_and_the_headline(self):
        html = render_html(shipping_readout())
        assert "SHIP" in html
        assert "+0.02" in html
        assert "practical threshold" in html

    def test_gate_appears_before_the_headline(self):
        """A reader who sees the effect first has already formed an opinion.

        Compares the *markup*, not bare substrings: class names also appear in the
        stylesheet above the body, so searching for "hero" alone found the CSS rule and
        made this assertion meaningless.
        """
        html = render_html(shipping_readout())
        assert html.index('<section class="gate">') < html.index('<section class="headline">')

    def test_synthetic_badge_is_present(self):
        assert "synthetic data" in render_html(shipping_readout())

    def test_blocked_readout_hides_every_metric_number(self):
        html = render_html(blocked_readout())
        assert "Blocked" in html
        assert "+0.02" not in html
        assert "wall, not a footnote" in html

    def test_moved_guardrail_is_flagged(self):
        readout = build_readout(
            spec(),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", -0.03, (-0.04, -0.02), 1e-8),
            },
        )
        assert "⚠" in render_html(readout)

    def test_escapes_untrusted_text(self):
        readout = build_readout(
            spec(),
            FAILING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.9),
            },
            override_reason="<script>alert('xss')</script>",
        )
        html = render_html(readout)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_custom_title(self):
        assert "My Readout" in render_html(shipping_readout(), title="My Readout")


class TestRenderMarkdown:
    def test_includes_decision_and_metrics(self):
        md = render_markdown(shipping_readout())
        assert "SHIP" in md
        assert "`retention_7`" in md
        assert "|---" in md

    def test_blocked_readout_hides_numbers(self):
        md = render_markdown(blocked_readout())
        assert "BLOCKED" in md
        assert "+0.02" not in md
        assert "wall, not a footnote" in md

    def test_records_an_override(self):
        readout = build_readout(
            spec(),
            FAILING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.9),
            },
            override_reason="known outage",
        )
        assert "Override recorded" in render_markdown(readout)

    def test_notes_the_guardrail_threshold_limitation(self):
        assert "PRD O5" in render_markdown(shipping_readout())

    def test_provenance_badge(self):
        assert "synthetic data" in render_markdown(shipping_readout())
