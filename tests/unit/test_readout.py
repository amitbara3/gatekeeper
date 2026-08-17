"""The decision layer: ship / hold / inconclusive / blocked."""

from __future__ import annotations

import pytest

from gatekeeper.report.readout import build_readout
from gatekeeper.spec import ExperimentSpec
from gatekeeper.types import (
    DataSource,
    Decision,
    EffectEstimate,
    Estimand,
    SanityCheck,
    SanityReport,
    SpecViolation,
)

PASSING = SanityReport(
    checks=(
        SanityCheck("srm", True, "split is fine"),
        SanityCheck("unique_units", True, "all distinct"),
    )
)
FAILING = SanityReport(
    checks=(
        SanityCheck("srm", False, "SAMPLE RATIO MISMATCH: 51.1/48.9"),
        SanityCheck("unique_units", True, "all distinct"),
    )
)


def spec(**overrides) -> ExperimentSpec:
    base = {
        "name": "test",
        "dataset": "cookie_cats",
        "registered_on": "2026-08-17",
        "primary_metric": "retention_7",
        "direction": "higher_is_better",
        "guardrail_metrics": ("retention_1",),
        "mde": 0.0075,
        "practical_threshold": 0.01,
        "expected_shares": {"gate_30": 0.5, "gate_40": 0.5},
    }
    base.update(overrides)
    return ExperimentSpec(**base)  # type: ignore[arg-type]


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


class TestBlocked:
    def test_failing_sanity_blocks_and_reports_no_metrics(self):
        r = build_readout(
            spec(),
            FAILING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", 0.0, (-0.005, 0.005), 0.9),
            },
        )
        assert r.decision is Decision.BLOCKED
        assert r.is_blocked
        assert r.metrics == ()
        assert "SAMPLE RATIO MISMATCH" not in r.rationale  # names the check, not the detail
        assert "srm" in r.rationale

    def test_blocked_readout_hides_metric_numbers_in_the_render(self):
        """A blocked result must be a wall, not a footnote (Design §4.2)."""
        r = build_readout(
            spec(),
            FAILING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", 0.0, (-0.005, 0.005), 0.9),
            },
        )
        text = r.render_text()
        assert "BLOCKED" in text
        assert "+0.02" not in text

    def test_override_unblocks_and_is_recorded(self):
        r = build_readout(
            spec(),
            FAILING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", 0.0, (-0.005, 0.005), 0.9),
            },
            override_reason="known logging outage, ticket 123",
        )
        assert r.decision is not Decision.BLOCKED
        assert r.override_reason == "known logging outage, ticket 123"
        assert "OVERRIDE RECORDED" in r.render_text()


class TestShip:
    def test_clear_improvement_with_quiet_guardrails_ships(self):
        r = build_readout(
            spec(),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", 0.001, (-0.004, 0.006), 0.7),
            },
        )
        assert r.decision is Decision.SHIP
        assert r.primary.practically_significant
        assert r.moved_guardrails == ()

    def test_a_moved_guardrail_downgrades_a_ship_to_hold(self):
        """A guardrail moving without explanation is what guardrails are for."""
        r = build_readout(
            spec(),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", -0.03, (-0.04, -0.02), 1e-8),
            },
        )
        assert r.decision is Decision.HOLD
        assert "guardrail" in r.rationale
        assert len(r.moved_guardrails) == 1

    def test_lower_is_better_flips_the_direction(self):
        r = build_readout(
            spec(direction="lower_is_better"),
            PASSING,
            {
                "retention_7": estimate("retention_7", -0.02, (-0.025, -0.015), 1e-9),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.9),
            },
        )
        assert r.decision is Decision.SHIP
        assert r.primary.direction == "improvement"


class TestHold:
    def test_practically_significant_regression_holds(self):
        r = build_readout(
            spec(),
            PASSING,
            {
                "retention_7": estimate("retention_7", -0.02, (-0.025, -0.015), 1e-9),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.9),
            },
        )
        assert r.decision is Decision.HOLD
        assert r.primary.direction == "regression"
        assert "harmful direction" in r.rationale


class TestInconclusive:
    def test_statistically_significant_but_practically_irrelevant(self):
        """The case R1.4 exists for: a real but meaningless effect."""
        r = build_readout(
            spec(),
            PASSING,
            {
                # Tight interval, excludes zero, entirely inside +/-0.01.
                "retention_7": estimate("retention_7", 0.004, (0.002, 0.006), 1e-4),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.9),
            },
        )
        assert r.decision is Decision.INCONCLUSIVE
        assert r.primary.statistically_significant
        assert not r.primary.practically_significant
        assert "ruled out" in r.rationale

    def test_informative_null_says_so(self):
        r = build_readout(
            spec(),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.0005, (-0.003, 0.004), 0.8),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.9),
            },
        )
        assert r.decision is Decision.INCONCLUSIVE
        assert "ruled out" in r.rationale
        assert "informative null" in r.rationale

    def test_underpowered_result_is_distinguished_from_an_informative_null(self):
        """ "Not significant" must not be conflated with "no effect" (R1.4)."""
        r = build_readout(
            spec(),
            PASSING,
            {
                # Wide interval: cannot confirm OR exclude a meaningful effect.
                "retention_7": estimate("retention_7", 0.005, (-0.05, 0.06), 0.6),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.9),
            },
        )
        assert r.decision is Decision.INCONCLUSIVE
        assert "lacked the precision" in r.rationale
        assert "ruled out" not in r.rationale

    def test_moved_guardrail_is_noted_even_when_inconclusive(self):
        r = build_readout(
            spec(),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.001, (-0.003, 0.005), 0.8),
                "retention_1": estimate("retention_1", -0.03, (-0.04, -0.02), 1e-8),
            },
        )
        assert r.decision is Decision.INCONCLUSIVE
        assert "guardrail" in r.rationale


class TestFamilyEnforcement:
    def test_missing_declared_metric_raises(self):
        with pytest.raises(SpecViolation, match="no estimate supplied"):
            build_readout(
                spec(), PASSING, {"retention_7": estimate("retention_7", 0.02, (0.01, 0.03), 0.01)}
            )

    def test_undeclared_metric_raises(self):
        with pytest.raises(SpecViolation, match="undeclared metric"):
            build_readout(
                spec(),
                PASSING,
                {
                    "retention_7": estimate("retention_7", 0.02, (0.01, 0.03), 0.01),
                    "retention_1": estimate("retention_1", 0.0, (-0.01, 0.01), 0.9),
                    "dau": estimate("dau", 0.0, (-0.01, 0.01), 0.9),
                },
            )

    def test_mixed_provenance_raises(self):
        real = EffectEstimate(
            estimand=Estimand(outcome="retention_1", treatment="version"),
            point=0.0,
            ci=(-0.01, 0.01),
            method="two_proportion_z",
            assumptions=("x",),
            data_source=DataSource.REAL,
            n_per_arm={"gate_30": 10, "gate_40": 10},
            p_value=0.9,
        )
        with pytest.raises(ValueError, match="mix data provenance"):
            build_readout(
                spec(),
                PASSING,
                {
                    "retention_7": estimate("retention_7", 0.02, (0.01, 0.03), 0.01),
                    "retention_1": real,
                },
            )


class TestMultiplicity:
    def test_correction_can_flip_a_significant_metric_to_non_significant(self):
        """The reason multiplicity correction exists, visible in a decision.

        Raw p=0.03 would reject on its own. In a declared family of three it becomes
        0.03 * 3/1 = 0.09 and does not. That change is the correction doing its job.
        """
        r = build_readout(
            spec(guardrail_metrics=("retention_1", "sum_gamerounds")),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 0.03),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.50),
                "sum_gamerounds": estimate("sum_gamerounds", 1.0, (-1.0, 3.0), 0.90),
            },
        )
        assert r.primary.adjusted_p == pytest.approx(0.09)
        assert not r.primary.statistically_significant
        assert r.moved_guardrails == ()

    def test_tied_p_values_inherit_the_least_conservative_adjustment(self):
        """BH step-up behaviour, which is easy to get backwards.

        m=3 with every raw p=0.02 gives per-rank values [0.06, 0.03, 0.02]; the
        right-to-left cumulative minimum pulls all three down to 0.02, so all three
        reject. The adjusted value for rank i is ``min over j>=i``, so a tie inherits
        the smallest -- not the largest.
        """
        r = build_readout(
            spec(guardrail_metrics=("retention_1", "sum_gamerounds")),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 0.02),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.02),
                "sum_gamerounds": estimate("sum_gamerounds", 1.0, (-1.0, 3.0), 0.02),
            },
        )
        for m in r.metrics:
            assert m.adjusted_p == pytest.approx(0.02)
            assert m.statistically_significant

    def test_estimator_without_a_p_value_is_handled(self):
        no_p = EffectEstimate(
            estimand=Estimand(outcome="retention_1", treatment="version"),
            point=0.0,
            ci=(-0.01, 0.01),
            method="bootstrap_bca",
            assumptions=("x",),
            data_source=DataSource.SYNTHETIC,
            n_per_arm={"gate_30": 10, "gate_40": 10},
            p_value=None,
        )
        r = build_readout(
            spec(),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": no_p,
            },
        )
        guardrail = r.guardrails[0]
        assert guardrail.adjusted_p is None
        assert not guardrail.statistically_significant


class TestRender:
    def test_render_includes_the_essentials(self):
        r = build_readout(
            spec(),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.9),
            },
        )
        text = r.render_text()
        assert "SHIP" in text
        assert "retention_7" in text
        assert "practical threshold" in text
        assert "PRIMARY" in text

    def test_synthetic_badge_is_shown(self):
        r = build_readout(
            spec(),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.9),
            },
        )
        assert "[SYNTHETIC DATA]" in r.render_text()
        assert r.is_synthetic

    def test_primary_accessor_and_guardrails(self):
        r = build_readout(
            spec(),
            PASSING,
            {
                "retention_7": estimate("retention_7", 0.02, (0.015, 0.025), 1e-9),
                "retention_1": estimate("retention_1", 0.0, (-0.004, 0.004), 0.9),
            },
        )
        assert r.primary.metric == "retention_7"
        assert [g.metric for g in r.guardrails] == ["retention_1"]
