"""Schema validation: strict typing, loud failure, no silent repair."""

from __future__ import annotations

import pandas as pd
import pytest

from gatekeeper.data.schema import COOKIE_CATS, ColumnSpec, DatasetSchema, ExperimentData, validate
from gatekeeper.types import DataSource, PostTreatmentCovariateError, SchemaViolation


class TestValidate:
    def test_valid_frame_passes_and_is_typed(self, raw_frame: pd.DataFrame):
        out = validate(raw_frame, COOKIE_CATS)
        assert out["userid"].dtype == "int64"
        assert out["retention_1"].dtype == bool
        assert out["sum_gamerounds"].dtype == "int64"

    def test_missing_column_raises(self, raw_frame: pd.DataFrame):
        with pytest.raises(SchemaViolation, match="missing required column"):
            validate(raw_frame.drop(columns=["retention_7"]), COOKIE_CATS)

    def test_empty_frame_raises(self, raw_frame: pd.DataFrame):
        with pytest.raises(SchemaViolation, match="no rows"):
            validate(raw_frame.iloc[0:0], COOKIE_CATS)

    def test_unexpected_variant_label_raises(self, raw_frame: pd.DataFrame):
        bad = raw_frame.copy()
        bad.loc[0, "version"] = "gate_50"
        with pytest.raises(SchemaViolation, match="unexpected value"):
            validate(bad, COOKIE_CATS)

    def test_null_in_numeric_column_raises_rather_than_imputing(self, raw_frame: pd.DataFrame):
        bad = raw_frame.copy()
        bad.loc[1, "sum_gamerounds"] = None
        with pytest.raises(SchemaViolation, match="null value"):
            validate(bad, COOKIE_CATS)

    def test_non_integral_value_in_int_column_raises(self, raw_frame: pd.DataFrame):
        bad = raw_frame.copy()
        bad["sum_gamerounds"] = bad["sum_gamerounds"].astype(float)
        bad.loc[0, "sum_gamerounds"] = 3.5
        with pytest.raises(SchemaViolation, match="non-integral"):
            validate(bad, COOKIE_CATS)

    def test_string_booleans_are_parsed(self, raw_frame: pd.DataFrame):
        as_str = raw_frame.copy()
        as_str["retention_1"] = ["True", "False", "TRUE", "false", "0", "1"]
        out = validate(as_str, COOKIE_CATS)
        assert out["retention_1"].tolist() == [True, False, True, False, False, True]

    def test_unrecognised_boolean_token_raises(self, raw_frame: pd.DataFrame):
        bad = raw_frame.copy()
        bad["retention_1"] = ["True", "False", "maybe", "False", "True", "False"]
        with pytest.raises(SchemaViolation, match="unrecognised boolean"):
            validate(bad, COOKIE_CATS)

    def test_extra_columns_are_dropped_without_error(self, raw_frame: pd.DataFrame):
        extra = raw_frame.assign(country="SE")
        out = validate(extra, COOKIE_CATS)
        assert "country" not in out.columns

    def test_duplicate_units_do_not_raise_here(self, raw_frame: pd.DataFrame):
        """Uniqueness is a sanity-gate concern, not a parse concern.

        It must surface as a blockable SanityReport with an actionable message
        (Design §4.2), not as an ingest crash.
        """
        dupe = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
        out = validate(dupe, COOKIE_CATS)
        assert len(out) == 7


class TestDatasetSchema:
    def test_variants_puts_control_first(self):
        assert COOKIE_CATS.variants[0] == "gate_30"
        assert set(COOKIE_CATS.variants) == {"gate_30", "gate_40"}

    def test_post_treatment_columns_are_declared(self):
        assert "sum_gamerounds" in COOKIE_CATS.post_treatment_columns
        assert "userid" not in COOKIE_CATS.post_treatment_columns

    def test_metric_columns_exclude_keys(self):
        assert "userid" not in COOKIE_CATS.metric_columns
        assert "version" not in COOKIE_CATS.metric_columns
        assert "retention_7" in COOKIE_CATS.metric_columns

    def test_unknown_column_lookup_raises(self):
        with pytest.raises(KeyError, match="no column"):
            COOKIE_CATS.column("nope")

    def test_variant_column_without_allowed_values_is_rejected(self):
        with pytest.raises(ValueError, match="must declare allowed_values"):
            DatasetSchema(
                name="bad",
                unit_col="uid",
                variant_col="arm",
                control="c",
                columns=(ColumnSpec("uid", "int"), ColumnSpec("arm", "str")),
            )

    def test_control_must_be_an_allowed_variant(self):
        with pytest.raises(ValueError, match="not among the allowed variants"):
            DatasetSchema(
                name="bad",
                unit_col="uid",
                variant_col="arm",
                control="missing",
                columns=(
                    ColumnSpec("uid", "int"),
                    ColumnSpec("arm", "str", allowed_values=frozenset({"a", "b"})),
                ),
            )

    def test_schema_must_spec_its_own_key_columns(self):
        with pytest.raises(ValueError, match="no spec for it"):
            DatasetSchema(
                name="bad",
                unit_col="absent",
                variant_col="arm",
                control="a",
                columns=(ColumnSpec("arm", "str", allowed_values=frozenset({"a", "b"})),),
            )


class TestExperimentData:
    def test_n_per_arm(self, tiny: ExperimentData):
        assert tiny.n_per_arm == {"gate_30": 3, "gate_40": 3}

    def test_control_and_treatment_accessors(self, tiny: ExperimentData):
        assert tiny.control == "gate_30"
        assert tiny.treatment == "gate_40"

    def test_arm_returns_only_that_arm(self, tiny: ExperimentData):
        assert set(tiny.arm("gate_40")["version"]) == {"gate_40"}

    def test_unknown_arm_raises(self, tiny: ExperimentData):
        with pytest.raises(KeyError, match="no arm"):
            tiny.arm("gate_99")

    def test_outcome_returns_float_array(self, tiny: ExperimentData):
        vals = tiny.outcome("retention_7", "gate_30")
        assert vals.dtype == float
        assert vals.tolist() == [0.0, 0.0, 1.0]

    def test_unknown_metric_raises(self, tiny: ExperimentData):
        with pytest.raises(KeyError, match="no column"):
            tiny.outcome("nope", "gate_30")

    def test_treatment_property_raises_for_multi_arm(self):
        schema = DatasetSchema(
            name="abn",
            unit_col="uid",
            variant_col="arm",
            control="a",
            columns=(
                ColumnSpec("uid", "int"),
                ColumnSpec("arm", "str", allowed_values=frozenset({"a", "b", "c"})),
                ColumnSpec("y", "float"),
            ),
        )
        df = pd.DataFrame({"uid": [1, 2, 3], "arm": ["a", "b", "c"], "y": [1.0, 2.0, 3.0]})
        data = ExperimentData.from_frame(df, schema=schema, data_source=DataSource.SYNTHETIC)
        with pytest.raises(ValueError, match="exactly one treatment arm"):
            _ = data.treatment

    def test_variants_reflects_arms_actually_present(self, raw_frame: pd.DataFrame):
        one_arm = raw_frame[raw_frame["version"] == "gate_30"]
        data = ExperimentData.from_frame(one_arm, data_source=DataSource.SYNTHETIC)
        assert data.variants == ("gate_30",)


class TestConstructionIsGuarded:
    """Direct construction must not smuggle in an unvalidated frame.

    ``verify_conforms`` runs in ``__post_init__``, so the "always use from_frame"
    convention is enforced rather than merely documented -- and a corrupted Parquet
    cache is caught on read instead of feeding wrong dtypes to an estimator.
    """

    def test_raw_unvalidated_frame_is_rejected(self, raw_frame: pd.DataFrame):
        # Strings where booleans belong: exactly what an unparsed CSV looks like.
        unparsed = raw_frame.copy()
        unparsed["retention_1"] = ["True", "False", "True", "True", "False", "False"]
        with pytest.raises(SchemaViolation, match="unexpected dtype"):
            ExperimentData(frame=unparsed, schema=COOKIE_CATS, data_source=DataSource.SYNTHETIC)

    def test_error_names_the_right_constructor(self, raw_frame: pd.DataFrame):
        unparsed = raw_frame.copy()
        unparsed["sum_gamerounds"] = unparsed["sum_gamerounds"].astype(str)
        with pytest.raises(SchemaViolation, match="from_frame"):
            ExperimentData(frame=unparsed, schema=COOKIE_CATS, data_source=DataSource.SYNTHETIC)

    def test_missing_column_is_rejected(self, raw_frame: pd.DataFrame):
        validated = validate(raw_frame, COOKIE_CATS).drop(columns=["retention_7"])
        with pytest.raises(SchemaViolation, match="missing column"):
            ExperimentData(frame=validated, schema=COOKIE_CATS, data_source=DataSource.SYNTHETIC)

    def test_corrupt_cache_message_mentions_the_cache(self, raw_frame: pd.DataFrame):
        corrupt = validate(raw_frame, COOKIE_CATS)
        corrupt["retention_7"] = corrupt["retention_7"].astype(int)
        with pytest.raises(SchemaViolation, match="cache is corrupt"):
            ExperimentData(frame=corrupt, schema=COOKIE_CATS, data_source=DataSource.SYNTHETIC)

    def test_an_already_validated_frame_is_accepted(self, raw_frame: pd.DataFrame):
        validated = validate(raw_frame, COOKIE_CATS)
        data = ExperimentData(frame=validated, schema=COOKIE_CATS, data_source=DataSource.SYNTHETIC)
        assert len(data.frame) == 6

    def test_from_frame_still_coerces_happily(self, raw_frame: pd.DataFrame):
        as_str = raw_frame.copy()
        as_str["retention_1"] = ["True", "False", "TRUE", "false", "0", "1"]
        data = ExperimentData.from_frame(as_str, data_source=DataSource.SYNTHETIC)
        assert data.frame["retention_1"].dtype == bool


class TestPostTreatmentGuard:
    """R1.7 -- the most tempting mistake available in this repo."""

    def test_post_treatment_covariate_is_rejected(self, tiny: ExperimentData):
        with pytest.raises(PostTreatmentCovariateError, match="after treatment assignment"):
            tiny.assert_pre_treatment("sum_gamerounds")

    def test_retention_is_also_post_treatment(self, tiny: ExperimentData):
        with pytest.raises(PostTreatmentCovariateError):
            tiny.assert_pre_treatment("retention_7")

    def test_error_message_explains_why(self, tiny: ExperimentData):
        with pytest.raises(PostTreatmentCovariateError, match="mediator"):
            tiny.assert_pre_treatment("sum_gamerounds")

    def test_pre_treatment_column_is_allowed(self, tiny: ExperimentData):
        tiny.assert_pre_treatment("userid")  # must not raise
