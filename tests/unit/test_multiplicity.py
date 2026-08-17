"""Multiplicity correction, with hand-computed BH values."""

from __future__ import annotations

import pytest

from gatekeeper.frequentist.multiplicity import correct, correct_spec_metrics


class TestBonferroni:
    def test_hand_computed(self):
        r = correct([0.001, 0.5], method="bonferroni")
        assert r.adjusted == pytest.approx((0.002, 1.0))
        assert r.rejected == (True, False)

    def test_clamped_at_one(self):
        r = correct([0.4, 0.6, 0.8], method="bonferroni")
        assert all(p <= 1.0 for p in r.adjusted)
        assert r.adjusted == pytest.approx((1.0, 1.0, 1.0))

    def test_single_test_is_unchanged(self):
        r = correct([0.03], method="bonferroni")
        assert r.adjusted == pytest.approx((0.03,))
        assert r.rejected == (True,)


class TestBenjaminiHochberg:
    def test_hand_computed_borderline_family(self):
        """p = [.01,.02,.03,.04,.05], m=5: every raw BH value is exactly 0.05."""
        r = correct([0.01, 0.02, 0.03, 0.04, 0.05])
        assert r.adjusted == pytest.approx((0.05, 0.05, 0.05, 0.05, 0.05))
        assert all(r.rejected)
        assert r.n_rejected == 5

    def test_hand_computed_two_test_family(self):
        # sorted [0.001, 0.5]: 0.001*2/1 = 0.002 ; 0.5*2/2 = 0.5
        r = correct([0.001, 0.5])
        assert r.adjusted == pytest.approx((0.002, 0.5))
        assert r.rejected == (True, False)

    def test_input_order_is_preserved(self):
        # unsorted [0.04, 0.01], m=2 -> sorted [0.01, 0.04]
        # 0.01*2/1 = 0.02 ; 0.04*2/2 = 0.04  -> monotone already
        r = correct([0.04, 0.01])
        assert r.adjusted == pytest.approx((0.04, 0.02))

    def test_adjusted_values_are_monotone_in_the_raw_values(self):
        """A more significant test can never get a larger adjusted p-value."""
        raw = [0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.9]
        r = correct(raw)
        pairs = sorted(zip(raw, r.adjusted, strict=True))
        adjusted_in_raw_order = [a for _, a in pairs]
        assert adjusted_in_raw_order == sorted(adjusted_in_raw_order)

    def test_bh_is_never_more_conservative_than_bonferroni(self):
        raw = [0.001, 0.01, 0.02, 0.03, 0.2]
        bh = correct(raw)
        bonf = correct(raw, method="bonferroni")
        for a, b in zip(bh.adjusted, bonf.adjusted, strict=True):
            assert a <= b + 1e-12

    def test_bh_rejects_at_least_as_much_as_bonferroni(self):
        raw = [0.001, 0.01, 0.02, 0.03, 0.2]
        assert correct(raw).n_rejected >= correct(raw, method="bonferroni").n_rejected

    def test_all_null_family_rejects_nothing(self):
        r = correct([0.4, 0.5, 0.6, 0.99])
        assert r.n_rejected == 0

    def test_ties_are_handled(self):
        """Tied p-values all receive the least conservative adjustment.

        m=3, all p=0.02. Raw BH values by rank are [0.06, 0.03, 0.02], and the
        right-to-left cumulative minimum pulls every one down to 0.02. That is standard
        step-up behaviour, not a bug: the adjusted value for rank i is
        ``min over j>=i`` of the raw values, so a tie inherits the smallest.

        Sanity check by the rejection rule: the largest k with p_k <= (k/3)(0.05) is
        k=3 (0.02 <= 0.05), so all three are rejected -- consistent with adjusted=0.02.
        """
        r = correct([0.02, 0.02, 0.02])
        assert r.adjusted == pytest.approx((0.02, 0.02, 0.02))
        assert all(r.rejected)


class TestNoCorrection:
    def test_returns_raw_values(self):
        r = correct([0.01, 0.2], method="none")
        assert r.adjusted == pytest.approx((0.01, 0.2))
        assert r.rejected == (True, False)
        assert r.method == "none"


class TestValidation:
    def test_empty_family_raises(self):
        with pytest.raises(ValueError, match="nothing to correct"):
            correct([])

    def test_out_of_range_p_value_raises(self):
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            correct([0.5, 1.5])

    def test_nan_raises(self):
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            correct([0.5, float("nan")])

    def test_bad_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            correct([0.01], alpha=0.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown correction method"):
            correct([0.01], method="holm")  # type: ignore[arg-type]

    def test_boundary_p_values_are_allowed(self):
        r = correct([0.0, 1.0])
        assert r.adjusted[0] == pytest.approx(0.0)


class TestSpecFamilyEnforcement:
    """The correction is meaningless if the family is not the pre-declared one (R1.8)."""

    FAMILY = ("retention_7", "retention_1", "sum_gamerounds")

    def test_happy_path(self):
        out = correct_spec_metrics(
            {"retention_7": 0.002, "retention_1": 0.30, "sum_gamerounds": 0.60},
            self.FAMILY,
        )
        assert set(out) == set(self.FAMILY)
        assert out["retention_7"][1] is True
        assert out["retention_1"][1] is False

    def test_extra_metric_raises(self):
        with pytest.raises(ValueError, match="not in the declared family"):
            correct_spec_metrics(
                {
                    "retention_7": 0.01,
                    "retention_1": 0.2,
                    "sum_gamerounds": 0.3,
                    "dau": 0.04,
                },
                self.FAMILY,
            )

    def test_missing_metric_raises(self):
        """Dropping a declared metric shrinks the divisor and overstates significance."""
        with pytest.raises(ValueError, match="have no p-value"):
            correct_spec_metrics({"retention_7": 0.01}, self.FAMILY)

    def test_matches_the_positional_api(self):
        p = {"retention_7": 0.002, "retention_1": 0.30, "sum_gamerounds": 0.60}
        by_name = correct_spec_metrics(p, self.FAMILY)
        positional = correct([p[m] for m in self.FAMILY])
        for i, m in enumerate(self.FAMILY):
            assert by_name[m][0] == pytest.approx(positional.adjusted[i])

    def test_works_with_a_real_spec(self):
        from gatekeeper.data.ingest import project_root
        from gatekeeper.spec import load_spec

        spec = load_spec(project_root() / "specs" / "cookie_cats_gate.yaml")
        out = correct_spec_metrics(
            dict.fromkeys(spec.all_metrics, 0.01),
            spec.all_metrics,
            alpha=spec.alpha,
            method=spec.multiplicity_method,
        )
        assert len(out) == spec.n_metrics
