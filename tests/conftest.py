"""Shared fixtures.

Everything here is synthetic and seeded (R4.2). The real Cookie Cats CSV is not
required to run the suite -- see data/README.md.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gatekeeper.data.schema import COOKIE_CATS, ExperimentData
from gatekeeper.data.synthetic import SyntheticExperiment, make_cookie_cats_like
from gatekeeper.spec import ExperimentSpec
from gatekeeper.types import DataSource


@pytest.fixture
def synth() -> SyntheticExperiment:
    """A moderate synthetic experiment with a known non-zero effect."""
    return make_cookie_cats_like(n=20_000, seed=7)


@pytest.fixture
def data(synth: SyntheticExperiment) -> ExperimentData:
    return synth.data


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A tiny, hand-written frame matching the Cookie Cats schema."""
    return pd.DataFrame(
        {
            "userid": [1, 2, 3, 4, 5, 6],
            "version": ["gate_30", "gate_30", "gate_30", "gate_40", "gate_40", "gate_40"],
            "sum_gamerounds": [3, 40, 7, 120, 0, 15],
            "retention_1": [True, False, True, True, False, False],
            "retention_7": [False, False, True, True, False, False],
        }
    )


@pytest.fixture
def tiny(raw_frame: pd.DataFrame) -> ExperimentData:
    return ExperimentData.from_frame(
        raw_frame, schema=COOKIE_CATS, data_source=DataSource.SYNTHETIC
    )


@pytest.fixture
def spec() -> ExperimentSpec:
    """A valid spec mirroring specs/cookie_cats_gate.yaml."""
    return ExperimentSpec(
        name="test_spec",
        dataset="cookie_cats",
        registered_on="2026-08-17",
        primary_metric="retention_7",
        direction="higher_is_better",
        guardrail_metrics=("retention_1", "sum_gamerounds"),
        mde=0.0075,
        practical_threshold=0.01,
        expected_shares={"gate_30": 0.5, "gate_40": 0.5},
    )
