"""Assignment-integrity checks and the sanity gate."""

from __future__ import annotations

import pandas as pd
import pytest

from gatekeeper.checks.integrity import (
    check_arm_sizes,
    check_no_cross_arm_units,
    check_unique_units,
    run_sanity_checks,
)
from gatekeeper.data.schema import ExperimentData
from gatekeeper.data.synthetic import make_cookie_cats_like
from gatekeeper.types import DataSource, SanityCheckFailure


class TestUniqueUnits:
    def test_clean_data_passes(self, tiny: ExperimentData):
        check = check_unique_units(tiny)
        assert check.passed
        assert check.statistic == 0.0

    def test_duplicate_unit_fails(self, raw_frame: pd.DataFrame):
        dupe = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
        data = ExperimentData.from_frame(dupe, data_source=DataSource.SYNTHETIC)
        check = check_unique_units(data)
        assert not check.passed
        assert check.statistic == 1.0
        assert "independence" in check.detail

    def test_counts_multiple_duplicates(self, raw_frame: pd.DataFrame):
        dupe = pd.concat([raw_frame, raw_frame.iloc[[0, 1, 2]]], ignore_index=True)
        data = ExperimentData.from_frame(dupe, data_source=DataSource.SYNTHETIC)
        assert check_unique_units(data).statistic == 3.0


class TestCrossArmUnits:
    def test_clean_data_passes(self, tiny: ExperimentData):
        assert check_no_cross_arm_units(tiny).passed

    def test_unit_in_both_arms_fails(self, raw_frame: pd.DataFrame):
        leaked = raw_frame.copy()
        # userid 1 now appears in both arms.
        extra = raw_frame.iloc[[0]].copy()
        extra["version"] = "gate_40"
        leaked = pd.concat([leaked, extra], ignore_index=True)
        data = ExperimentData.from_frame(leaked, data_source=DataSource.SYNTHETIC)

        check = check_no_cross_arm_units(data)
        assert not check.passed
        assert check.statistic == 1.0
        assert "not sticky" in check.detail

    def test_duplicate_within_one_arm_is_not_cross_arm(self, raw_frame: pd.DataFrame):
        """A repeated row in a single arm is a uniqueness problem, not a leak."""
        dupe = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
        data = ExperimentData.from_frame(dupe, data_source=DataSource.SYNTHETIC)
        assert check_no_cross_arm_units(data).passed
        assert not check_unique_units(data).passed


class TestArmSizes:
    def test_adequate_arms_pass(self):
        exp = make_cookie_cats_like(n=2_000, seed=3)
        assert check_arm_sizes(exp.data, min_per_arm=100).passed

    def test_undersized_arm_fails(self, tiny: ExperimentData):
        check = check_arm_sizes(tiny, min_per_arm=100)
        assert not check.passed
        assert "below" in check.detail

    def test_missing_arm_fails(self, raw_frame: pd.DataFrame):
        one_arm = raw_frame[raw_frame["version"] == "gate_30"]
        data = ExperimentData.from_frame(one_arm, data_source=DataSource.SYNTHETIC)
        check = check_arm_sizes(data, min_per_arm=1)
        assert not check.passed
        assert "absent" in check.detail

    def test_rejects_nonsense_minimum(self, tiny: ExperimentData):
        with pytest.raises(ValueError, match="at least 1"):
            check_arm_sizes(tiny, min_per_arm=0)


class TestRunSanityChecks:
    def test_healthy_experiment_passes_every_check(self):
        exp = make_cookie_cats_like(n=20_000, seed=11)
        report = run_sanity_checks(exp.data)
        assert report.passed, report.summary()
        assert {c.name for c in report.checks} == {
            "srm",
            "unique_units",
            "no_cross_arm_units",
            "arm_sizes",
        }

    def test_gate_blocks_a_corrupted_experiment(self):
        """The gate must block, not warn (R1.3). Phase 3 exit criterion."""
        exp = make_cookie_cats_like(n=20_000, seed=11, treatment_share=0.45)
        report = run_sanity_checks(exp.data)
        assert not report.passed
        with pytest.raises(SanityCheckFailure, match="sanity check"):
            report.raise_if_failed()

    def test_all_checks_run_even_when_one_fails(self):
        """The report shows the full picture, not just the first problem."""
        exp = make_cookie_cats_like(n=20_000, seed=11, treatment_share=0.45)
        report = run_sanity_checks(exp.data)
        assert len(report.checks) == 4

    def test_override_is_recorded_rather_than_silent(self):
        exp = make_cookie_cats_like(n=20_000, seed=11, treatment_share=0.45)
        report = run_sanity_checks(exp.data)
        report.raise_if_failed(override_reason="deliberate imbalance for a test")

    def test_extra_checks_are_folded_in(self):
        from gatekeeper.checks.outliers import check_outlier_leverage

        exp = make_cookie_cats_like(n=5_000, seed=2)
        extra = check_outlier_leverage(exp.data, "sum_gamerounds", leverage_threshold=0.5)
        report = run_sanity_checks(exp.data, extra=[extra])
        assert any(c.name.startswith("outlier_leverage") for c in report.checks)

    def test_spec_threshold_is_honoured(self):
        exp = make_cookie_cats_like(n=20_000, seed=11, treatment_share=0.485)
        strict = run_sanity_checks(exp.data, srm_threshold=0.5)
        lenient = run_sanity_checks(exp.data, srm_threshold=1e-12)
        assert not strict.get("srm").passed
        assert lenient.get("srm").passed
