"""Data ingest, schema contracts, and synthetic generation."""

from __future__ import annotations

from gatekeeper.data.ingest import load_cookie_cats, project_root
from gatekeeper.data.schema import (
    COOKIE_CATS,
    ColumnSpec,
    DatasetSchema,
    ExperimentData,
    validate,
)
from gatekeeper.data.synthetic import (
    SyntheticExperiment,
    make_cookie_cats_like,
    make_null_experiment,
)

__all__ = [
    "COOKIE_CATS",
    "ColumnSpec",
    "DatasetSchema",
    "ExperimentData",
    "SyntheticExperiment",
    "load_cookie_cats",
    "make_cookie_cats_like",
    "make_null_experiment",
    "project_root",
    "validate",
]
