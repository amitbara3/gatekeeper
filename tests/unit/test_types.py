"""Tests for the core result types, especially their refusal to be constructed badly."""

from __future__ import annotations

import pytest

from gatekeeper.types import (
    DataSource,
    EffectEstimate,
    Estimand,
    EstimandTarget,
    SanityCheck,
    SanityCheckFailure,
    SanityReport,
    Scale,
)


def make_estimate(**overrides) -> EffectEstimate:
    kwargs = {
        "estimand": Estimand(outcome="retention_7", treatment="version"),
        "point": 0.02,
        "ci": (0.01, 0.03),
        "method": "two_proportion_z",
        "assumptions": ("independent units", "randomised assignment"),
        "data_source": DataSource.SYNTHETIC,
        "n_per_arm": {"gate_30": 100, "gate_40": 100},
    }
    kwargs.update(overrides)
    return EffectEstimate(**kwargs)  # type: ignore[arg-type]


class TestEstimand:
    def test_requires_outcome(self):
        with pytest.raises(ValueError, match="outcome"):
            Estimand(outcome="", treatment="version")

    def test_requires_treatment(self):
        with pytest.raises(ValueError, match="treatment"):
            Estimand(outcome="retention_7", treatment="")

    def test_defaults_to_absolute_ate(self):
        e = Estimand(outcome="retention_7", treatment="version")
        assert e.target is EstimandTarget.ATE
        assert e.scale is Scale.ABSOLUTE

    def test_describe_mentions_target_and_scale(self):
        e = Estimand(outcome="retention_7", treatment="version", target=EstimandTarget.LATE)
        assert "LATE" in e.describe()
        assert "retention_7" in e.describe()

    def test_is_frozen(self):
        e = Estimand(outcome="a", treatment="b")
        with pytest.raises(AttributeError):
            e.outcome = "c"  # type: ignore[misc]


class TestEffectEstimateValidation:
    def test_assumptions_are_mandatory(self):
        """R2.4 -- an estimate cannot exist without declaring what it assumes."""
        with pytest.raises(ValueError, match="no declared assumptions"):
            make_estimate(assumptions=())

    def test_inverted_ci_raises(self):
        with pytest.raises(ValueError, match="exceeds upper bound"):
            make_estimate(ci=(0.03, 0.01))

    def test_p_value_outside_unit_interval_raises(self):
        with pytest.raises(ValueError, match=r"p_value must be in \[0, 1\]"):
            make_estimate(p_value=1.5)

    def test_negative_se_raises(self):
        with pytest.raises(ValueError, match="se must be non-negative"):
            make_estimate(se=-0.1)

    def test_ci_level_must_be_a_probability(self):
        with pytest.raises(ValueError, match="ci_level must be in"):
            make_estimate(ci_level=95.0)

    def test_empty_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            make_estimate(method="")

    def test_empty_n_per_arm_raises(self):
        with pytest.raises(ValueError, match="at least one arm"):
            make_estimate(n_per_arm={})

    def test_degenerate_ci_is_allowed(self):
        """A zero-width interval is legal -- e.g. a finite-population exact value."""
        est = make_estimate(ci=(0.02, 0.02))
        assert est.ci_width == 0.0


class TestEffectEstimateInterpretation:
    def test_ci_excludes_zero(self):
        assert make_estimate(ci=(0.01, 0.03)).ci_excludes_zero
        assert make_estimate(ci=(-0.03, -0.01)).ci_excludes_zero
        assert not make_estimate(ci=(-0.01, 0.03)).ci_excludes_zero

    def test_practical_significance_requires_whole_ci_beyond_threshold(self):
        """The strict reading: we must be able to rule out effects below threshold."""
        # Interval entirely above +0.01
        assert make_estimate(point=0.02, ci=(0.015, 0.03)).is_practically_significant(0.01)
        # Interval entirely below -0.01
        assert make_estimate(point=-0.02, ci=(-0.03, -0.015)).is_practically_significant(0.01)
        # Point estimate exceeds threshold but interval overlaps the "does not matter"
        # band -- nothing has been established.
        assert not make_estimate(point=0.02, ci=(0.005, 0.04)).is_practically_significant(0.01)
        # Interval excludes zero but sits inside the band: statistically significant,
        # practically irrelevant. This is the case R1.4 exists for.
        assert not make_estimate(point=0.004, ci=(0.002, 0.006)).is_practically_significant(0.01)

    def test_practical_significance_rejects_negative_threshold(self):
        with pytest.raises(ValueError, match="non-negative"):
            make_estimate().is_practically_significant(-0.01)

    def test_excludes_effects_beyond(self):
        """The other half of an honest null result."""
        tight_null = make_estimate(point=0.0005, ci=(-0.002, 0.003))
        assert tight_null.excludes_effects_beyond(0.01)
        assert not tight_null.excludes_effects_beyond(0.001)

        wide_null = make_estimate(point=0.0, ci=(-0.05, 0.05))
        assert not wide_null.excludes_effects_beyond(0.01)

    def test_n_total_and_is_synthetic(self):
        est = make_estimate()
        assert est.n_total == 200
        assert est.is_synthetic
        assert not make_estimate(data_source=DataSource.REAL).is_synthetic

    def test_semi_synthetic_counts_as_synthetic(self):
        assert make_estimate(data_source=DataSource.SEMI_SYNTHETIC).is_synthetic

    def test_with_override_records_reason_and_preserves_fields(self):
        est = make_estimate(p_value=0.03, se=0.005, seed=42)
        stamped = est.with_override("SRM explained by a known logging outage")
        assert stamped.override_reason == "SRM explained by a known logging outage"
        assert stamped.point == est.point
        assert stamped.ci == est.ci
        assert stamped.p_value == est.p_value
        assert stamped.seed == 42
        assert est.override_reason is None  # original untouched

    def test_with_override_rejects_empty_reason(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_estimate().with_override("")

    def test_summary_formats_small_p_values(self):
        assert "<0.001" in make_estimate(p_value=1e-9).summary()
        assert "0.030" in make_estimate(p_value=0.03).summary()

    def test_summary_reports_provenance(self):
        assert "synthetic" in make_estimate().summary()


class TestSanityReport:
    def test_passing_report(self):
        r = SanityReport(checks=(SanityCheck("a", True, "fine"), SanityCheck("b", True, "fine")))
        assert r.passed
        assert r.failures == ()
        r.raise_if_failed()  # must not raise
        assert "PASSED" in r.summary()

    def test_failing_report_raises_with_all_failures_listed(self):
        r = SanityReport(
            checks=(
                SanityCheck("srm", False, "split is wrong"),
                SanityCheck("dupes", False, "duplicate units"),
                SanityCheck("sizes", True, "fine"),
            )
        )
        assert not r.passed
        assert len(r.failures) == 2
        with pytest.raises(SanityCheckFailure) as exc:
            r.raise_if_failed()
        assert "split is wrong" in str(exc.value)
        assert "duplicate units" in str(exc.value)

    def test_override_suppresses_the_raise(self):
        r = SanityReport(checks=(SanityCheck("srm", False, "split is wrong"),))
        r.raise_if_failed(override_reason="known outage, documented in ticket 123")

    def test_get_returns_named_check(self):
        r = SanityReport(checks=(SanityCheck("srm", True, "fine"),))
        assert r.get("srm").name == "srm"

    def test_get_missing_check_raises_and_lists_what_ran(self):
        r = SanityReport(checks=(SanityCheck("srm", True, "fine"),))
        with pytest.raises(KeyError, match="srm"):
            r.get("never_ran")

    def test_check_requires_a_detail_message(self):
        with pytest.raises(ValueError, match="detail message"):
            SanityCheck("srm", True, "")

    def test_check_rejects_invalid_p_value(self):
        with pytest.raises(ValueError, match=r"p_value must be in \[0, 1\]"):
            SanityCheck("srm", True, "fine", p_value=2.0)
