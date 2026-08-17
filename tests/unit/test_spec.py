"""The pre-registration spec -- validation and, more importantly, enforcement."""

from __future__ import annotations

import pytest

from gatekeeper.data.ingest import project_root
from gatekeeper.spec import ExperimentSpec, OutlierRule, load_spec
from gatekeeper.types import SpecViolation

BASE = {
    "name": "s",
    "dataset": "cookie_cats",
    "registered_on": "2026-08-17",
    "primary_metric": "retention_7",
    "direction": "higher_is_better",
    "mde": 0.0075,
    "practical_threshold": 0.01,
    "expected_shares": {"gate_30": 0.5, "gate_40": 0.5},
}


def spec_with(**overrides) -> ExperimentSpec:
    return ExperimentSpec(**{**BASE, **overrides})


class TestValidation:
    def test_minimal_valid_spec(self):
        s = spec_with()
        assert s.primary_metric == "retention_7"
        assert s.mode == "confirmatory"

    def test_primary_metric_cannot_also_be_a_guardrail(self):
        with pytest.raises(ValueError, match="one or the other"):
            spec_with(guardrail_metrics=("retention_7",))

    def test_duplicate_guardrails_rejected(self):
        with pytest.raises(ValueError, match="duplicate guardrail"):
            spec_with(guardrail_metrics=("retention_1", "retention_1"))

    def test_shares_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            spec_with(expected_shares={"gate_30": 0.5, "gate_40": 0.4})

    def test_shares_must_cover_two_arms(self):
        with pytest.raises(ValueError, match="at least two arms"):
            spec_with(expected_shares={"gate_30": 1.0})

    def test_negative_share_rejected(self):
        with pytest.raises(ValueError, match=r"sum to 1|positive"):
            spec_with(expected_shares={"gate_30": 1.5, "gate_40": -0.5})

    def test_mde_above_practical_threshold_is_rejected(self):
        """An underpowered plan cannot answer its own question."""
        with pytest.raises(ValueError, match="cannot reliably detect"):
            spec_with(mde=0.02, practical_threshold=0.01)

    def test_mde_equal_to_threshold_is_allowed(self):
        assert spec_with(mde=0.01, practical_threshold=0.01).mde == 0.01

    def test_alpha_must_be_a_probability(self):
        with pytest.raises(ValueError):
            spec_with(alpha=1.5)

    def test_unknown_field_is_rejected(self):
        """extra='forbid' -- a typo in a spec must not be silently ignored."""
        with pytest.raises(ValueError):
            spec_with(primry_metric="oops")

    def test_spec_is_frozen(self):
        s = spec_with()
        with pytest.raises(ValueError):
            s.primary_metric = "retention_1"  # type: ignore[misc]

    def test_duplicate_outlier_rules_rejected(self):
        with pytest.raises(ValueError, match="more than one outlier rule"):
            spec_with(
                outlier_rules=(
                    OutlierRule(metric="sum_gamerounds", method="none"),
                    OutlierRule(metric="sum_gamerounds", method="winsorize", percentile=99.9),
                )
            )


class TestOutlierRule:
    def test_none_method_takes_no_percentile(self):
        with pytest.raises(ValueError, match="takes no percentile"):
            OutlierRule(metric="m", method="none", percentile=99.0)

    def test_winsorize_requires_a_percentile(self):
        with pytest.raises(ValueError, match="needs a percentile"):
            OutlierRule(metric="m", method="winsorize")

    def test_percentile_must_be_in_range(self):
        with pytest.raises(ValueError, match=r"must be in \(50, 100\)"):
            OutlierRule(metric="m", method="winsorize", percentile=10.0)

    def test_describe_mentions_both_arms(self):
        rule = OutlierRule(metric="m", method="winsorize", percentile=99.9)
        assert "both arms" in rule.describe()

    def test_default_rule_is_no_trimming(self):
        assert spec_with().outlier_rule_for("sum_gamerounds").method == "none"

    def test_declared_rule_is_returned(self):
        s = spec_with(
            outlier_rules=(
                OutlierRule(metric="sum_gamerounds", method="winsorize", percentile=99.9),
            )
        )
        rule = s.outlier_rule_for("sum_gamerounds")
        assert rule.method == "winsorize"
        assert rule.percentile == 99.9


class TestEnforcement:
    """The reason this class exists is R1.2: the spec is not documentation."""

    def test_undeclared_metric_raises(self):
        s = spec_with(guardrail_metrics=("retention_1",))
        with pytest.raises(SpecViolation, match="metric fishing"):
            s.assert_metric_declared("sum_gamerounds")

    def test_declared_metrics_pass(self):
        s = spec_with(guardrail_metrics=("retention_1",))
        s.assert_metric_declared("retention_7")
        s.assert_metric_declared("retention_1")

    def test_promoting_a_secondary_metric_to_primary_raises(self):
        s = spec_with(guardrail_metrics=("retention_1",))
        with pytest.raises(SpecViolation, match="primary metric is fixed"):
            s.assert_primary("retention_1")

    def test_undeclared_subgroup_raises(self):
        with pytest.raises(SpecViolation, match="fishing"):
            spec_with().assert_subgroup_declared("country")

    def test_declared_subgroup_passes(self):
        spec_with(declared_subgroups=("country",)).assert_subgroup_declared("country")

    def test_fixed_horizon_spec_forbids_a_second_look(self):
        """R1.5 -- peeking without a correction."""
        s = spec_with(stopping_rule="fixed_horizon")
        s.assert_single_look_allowed(1)
        with pytest.raises(SpecViolation, match="inflate the false-positive rate"):
            s.assert_single_look_allowed(2)

    def test_sequential_spec_allows_repeated_looks(self):
        s = spec_with(stopping_rule="sequential")
        s.assert_single_look_allowed(7)

    def test_all_metrics_puts_primary_first(self):
        s = spec_with(guardrail_metrics=("retention_1", "sum_gamerounds"))
        assert s.all_metrics == ("retention_7", "retention_1", "sum_gamerounds")
        assert s.n_metrics == 3


class TestLoadSpec:
    def test_committed_spec_loads_and_validates(self):
        """The real spec file must stay valid -- it is the project's contract."""
        s = load_spec(project_root() / "specs" / "cookie_cats_gate.yaml")
        assert s.name == "cookie_cats_gate"
        assert s.primary_metric == "retention_7"
        assert s.mode == "confirmatory"
        assert s.stopping_rule == "fixed_horizon"
        assert s.srm_threshold == 0.0005
        assert s.outlier_rule_for("sum_gamerounds").method == "winsorize"
        assert s.all_metrics == ("retention_7", "retention_1", "sum_gamerounds")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_spec(tmp_path / "nope.yaml")

    def test_non_mapping_yaml_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(SpecViolation, match="must be a YAML mapping"):
            load_spec(p)

    def test_invalid_spec_content_raises_spec_violation(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: x\ndataset: cookie_cats\n", encoding="utf-8")
        with pytest.raises(SpecViolation, match="failed validation"):
            load_spec(p)

    def test_summary_is_readable(self):
        text = spec_with(guardrail_metrics=("retention_1",)).summary()
        assert "retention_7" in text
        assert "confirmatory" in text
