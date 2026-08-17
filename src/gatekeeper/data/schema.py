"""Column contracts and the typed experiment container.

Validation here is strict and fails loudly (Rules §5). We parse recognised
representations of a type -- ``"True"`` into ``True`` -- but never repair invalid
data. No ``fillna`` to make the code run; no dropping rows to make a check pass.

Each column carries a ``post_treatment`` flag. That single piece of schema metadata
is what lets the CUPED guard enforce R1.7 mechanically instead of relying on the
analyst to remember that ``sum_gamerounds`` is a mediator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from gatekeeper.types import DataSource, SchemaViolation

__all__ = [
    "COOKIE_CATS",
    "ColumnKind",
    "ColumnSpec",
    "DatasetSchema",
    "ExperimentData",
]

ColumnKind = Literal["int", "float", "bool", "str"]

_TRUE_TOKENS = frozenset({"true", "t", "1", "yes"})
_FALSE_TOKENS = frozenset({"false", "f", "0", "no"})


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Contract for a single column.

    Parameters
    ----------
    name
        Column name as it appears in the source file.
    kind
        Expected semantic type.
    allowed_values
        If set, any value outside this set raises. Used for variant labels, so a
        stray ``"gate_50"`` is an error rather than a silently ignored third arm.
    post_treatment
        ``True`` if this column is measured *after* treatment assignment takes
        effect. Post-treatment columns are mediators, not covariates, and adjusting
        for them biases a total-effect estimand (R1.7).
    unique
        If ``True``, duplicate values raise at ingest.
    description
        Human-readable note, surfaced in error messages.
    """

    name: str
    kind: ColumnKind
    allowed_values: frozenset[str] | None = None
    post_treatment: bool = False
    unique: bool = False
    """Declares that values *should* be unique.

    Enforced by :func:`gatekeeper.checks.integrity.check_unique_units`, **not** by
    :func:`validate`. The split is deliberate: schema validation answers "can this
    file be parsed and typed?", the sanity gate answers "is this experiment
    trustworthy?". Duplicated randomisation units are the second kind of problem, so
    they belong in a blockable :class:`~gatekeeper.types.SanityReport` with an
    actionable message (Design §4.2) rather than an ingest crash.
    """
    description: str = ""


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """The full contract for an experiment dataset.

    Parameters
    ----------
    name
        Identifier, e.g. ``"cookie_cats"``.
    columns
        One :class:`ColumnSpec` per expected column.
    unit_col
        The randomisation unit. Metrics must be measured at this grain, or variance
        needs a cluster-robust or delta-method correction (R1.13).
    variant_col
        The column holding variant labels.
    control
        Which variant label is the control arm.
    """

    name: str
    columns: tuple[ColumnSpec, ...]
    unit_col: str
    variant_col: str
    control: str

    def __post_init__(self) -> None:
        names = [c.name for c in self.columns]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate column specs in schema {self.name!r}")
        for required in (self.unit_col, self.variant_col):
            if required not in names:
                raise ValueError(
                    f"schema {self.name!r} declares {required!r} but has no spec for it"
                )
        variant_spec = self.column(self.variant_col)
        if variant_spec.allowed_values is None:
            raise ValueError(
                f"schema {self.name!r}: variant column {self.variant_col!r} must declare "
                "allowed_values, so an unexpected arm label raises instead of appearing "
                "as a silent third arm"
            )
        if self.control not in variant_spec.allowed_values:
            raise ValueError(
                f"control label {self.control!r} is not among the allowed variants "
                f"{sorted(variant_spec.allowed_values)}"
            )

    def column(self, name: str) -> ColumnSpec:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"schema {self.name!r} has no column {name!r}")

    @property
    def variants(self) -> tuple[str, ...]:
        """Allowed variant labels, control first."""
        allowed = self.column(self.variant_col).allowed_values
        assert allowed is not None  # guaranteed by __post_init__
        others = sorted(allowed - {self.control})
        return (self.control, *others)

    @property
    def metric_columns(self) -> tuple[str, ...]:
        """Columns usable as outcomes: everything but the unit and variant keys."""
        return tuple(
            c.name for c in self.columns if c.name not in (self.unit_col, self.variant_col)
        )

    @property
    def post_treatment_columns(self) -> frozenset[str]:
        """Columns that must never be used as covariates (R1.7)."""
        return frozenset(c.name for c in self.columns if c.post_treatment)

    @property
    def fingerprint(self) -> str:
        """Stable 12-char digest of this contract's content.

        Used in the Parquet cache filename so that **editing the schema invalidates
        the cache**. Without it, a cache hit is decided purely on file mtimes and a
        narrowed or retyped schema would silently return data validated against the
        old contract -- the cache holds validated data, so a hit skips revalidation
        entirely.

        Uses ``hashlib`` rather than ``hash()`` because the built-in is salted per
        process for strings, which would make the digest differ between runs.
        """
        parts = [self.name, self.unit_col, self.variant_col, self.control]
        for c in sorted(self.columns, key=lambda s: s.name):
            allowed = "" if c.allowed_values is None else "|".join(sorted(c.allowed_values))
            parts.append(f"{c.name}:{c.kind}:{allowed}:{c.post_treatment:d}:{c.unique:d}")
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# The Cookie Cats contract
# ---------------------------------------------------------------------------

COOKIE_CATS = DatasetSchema(
    name="cookie_cats",
    unit_col="userid",
    variant_col="version",
    control="gate_30",
    columns=(
        ColumnSpec(
            "userid",
            "int",
            unique=True,
            description="randomisation unit; must be unique",
        ),
        ColumnSpec(
            "version",
            "str",
            allowed_values=frozenset({"gate_30", "gate_40"}),
            description="gate_30 = control, gate_40 = treatment",
        ),
        ColumnSpec(
            "sum_gamerounds",
            "int",
            post_treatment=True,
            description=(
                "rounds played in the first 14 days; measured AFTER the player meets "
                "the gate, so it is a mediator and never a covariate (R1.7)"
            ),
        ),
        ColumnSpec("retention_1", "bool", post_treatment=True, description="returned day 1"),
        ColumnSpec("retention_7", "bool", post_treatment=True, description="returned day 7"),
    ),
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _parse_bool_column(series: pd.Series, name: str) -> pd.Series:
    """Parse a boolean column strictly, raising on unrecognised tokens."""
    if series.dtype == bool:
        return series
    if series.isna().any():
        raise SchemaViolation(
            f"column {name!r} contains {int(series.isna().sum())} null value(s); "
            "booleans must be fully populated (no imputation -- Rules §5)"
        )
    lowered = series.astype(str).str.strip().str.lower()
    unknown = set(lowered.unique()) - _TRUE_TOKENS - _FALSE_TOKENS
    if unknown:
        raise SchemaViolation(
            f"column {name!r} has unrecognised boolean value(s) {sorted(unknown)}; "
            f"expected one of {sorted(_TRUE_TOKENS | _FALSE_TOKENS)}"
        )
    return lowered.isin(_TRUE_TOKENS)


def _validate_numeric(series: pd.Series, spec: ColumnSpec) -> pd.Series:
    if series.isna().any():
        raise SchemaViolation(
            f"column {spec.name!r} contains {int(series.isna().sum())} null value(s); "
            "ingest does not impute (Rules §5)"
        )
    if spec.kind == "int":
        if not pd.api.types.is_numeric_dtype(series):
            raise SchemaViolation(f"column {spec.name!r} must be numeric, got dtype {series.dtype}")
        as_float = series.astype(float)
        if not np.all(np.equal(np.mod(as_float, 1), 0)):
            raise SchemaViolation(
                f"column {spec.name!r} is declared int but holds non-integral values"
            )
        return as_float.astype("int64")
    if not pd.api.types.is_numeric_dtype(series):
        raise SchemaViolation(f"column {spec.name!r} must be numeric, got dtype {series.dtype}")
    return series.astype(float)


def validate(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Validate ``df`` against ``schema``, returning a correctly typed copy.

    Raises
    ------
    SchemaViolation
        On a missing column, a wrong type, a null, an empty frame, or a variant
        label outside the declared set.

    Notes
    -----
    Extra columns are dropped with no error -- a source file may legitimately carry
    fields we do not model. Missing or malformed *declared* columns always raise.

    Uniqueness (``ColumnSpec.unique``) is **not** checked here; it is a sanity-gate
    concern, handled by :func:`gatekeeper.checks.integrity.check_unique_units`.
    """
    missing = [c.name for c in schema.columns if c.name not in df.columns]
    if missing:
        raise SchemaViolation(
            f"input is missing required column(s) {missing} for schema {schema.name!r}; "
            f"found {list(df.columns)}"
        )
    if len(df) == 0:
        raise SchemaViolation(f"input for schema {schema.name!r} has no rows")

    out = {}
    for spec in schema.columns:
        series = df[spec.name]
        if spec.kind == "bool":
            parsed = _parse_bool_column(series, spec.name)
        elif spec.kind in ("int", "float"):
            parsed = _validate_numeric(series, spec)
        else:
            if series.isna().any():
                raise SchemaViolation(f"column {spec.name!r} contains null value(s)")
            parsed = series.astype(str).str.strip()

        if spec.allowed_values is not None:
            seen = set(parsed.unique())
            unexpected = seen - set(spec.allowed_values)
            if unexpected:
                raise SchemaViolation(
                    f"column {spec.name!r} contains unexpected value(s) {sorted(unexpected)}; "
                    f"schema allows {sorted(spec.allowed_values)}. An unrecognised arm label "
                    "is an instrumentation error, not a new variant."
                )
        out[spec.name] = parsed.reset_index(drop=True)

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# ExperimentData
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExperimentData:
    """A validated experiment dataset plus its contract and provenance.

    Construct via :func:`from_frame` (which validates) or the loaders in
    :mod:`gatekeeper.data.ingest` -- never by assigning to ``frame`` directly.
    """

    frame: pd.DataFrame
    schema: DatasetSchema
    data_source: DataSource

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        schema: DatasetSchema = COOKIE_CATS,
        data_source: DataSource = DataSource.REAL,
    ) -> ExperimentData:
        """Validate ``df`` against ``schema`` and wrap it."""
        return cls(frame=validate(df, schema), schema=schema, data_source=data_source)

    # -- accessors -------------------------------------------------------------

    @property
    def variants(self) -> tuple[str, ...]:
        """Variant labels *present in the data*, control first."""
        present = set(self.frame[self.schema.variant_col].unique())
        ordered = [v for v in self.schema.variants if v in present]
        return tuple(ordered)

    @property
    def control(self) -> str:
        return self.schema.control

    @property
    def treatment(self) -> str:
        """The single non-control arm; raises if there is not exactly one."""
        others = [v for v in self.variants if v != self.control]
        if len(others) != 1:
            raise ValueError(
                f"expected exactly one treatment arm, found {others}; "
                "use .arm(label) explicitly for A/B/n designs"
            )
        return others[0]

    @property
    def n_per_arm(self) -> dict[str, int]:
        counts = self.frame[self.schema.variant_col].value_counts()
        return {v: int(counts.get(v, 0)) for v in self.variants}

    def arm(self, variant: str) -> pd.DataFrame:
        """Rows belonging to ``variant``."""
        if variant not in self.variants:
            raise KeyError(f"no arm {variant!r} in data; present: {self.variants}")
        return self.frame[self.frame[self.schema.variant_col] == variant]

    def outcome(self, metric: str, variant: str) -> np.ndarray:
        """The ``metric`` values for one arm, as a float array."""
        if metric not in self.frame.columns:
            raise KeyError(f"no column {metric!r}; available: {list(self.frame.columns)}")
        return self.arm(variant)[metric].to_numpy(dtype=float)

    def assert_pre_treatment(self, column: str) -> None:
        """Raise if ``column`` is post-treatment (R1.7).

        Called by anything that needs a genuine covariate -- CUPED, regression
        adjustment, propensity models.
        """
        from gatekeeper.types import PostTreatmentCovariateError

        if column in self.schema.post_treatment_columns:
            spec = self.schema.column(column)
            raise PostTreatmentCovariateError(
                f"{column!r} is measured after treatment assignment and cannot be used "
                f"as a covariate. {spec.description}. Adjusting for a mediator biases a "
                "total-effect estimand while appearing to reduce variance (R1.7)."
            )

    def summary(self) -> str:
        arms = "  ".join(f"{k}={v:,}" for k, v in self.n_per_arm.items())
        return (
            f"{self.schema.name} [{self.data_source}]  n={len(self.frame):,}  {arms}\n"
            f"metrics: {', '.join(self.schema.metric_columns)}"
        )
