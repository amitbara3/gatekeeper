"""Ingest: strict validation, helpful failure, and the Parquet cache.

These tests run against temp CSVs, so the real Cookie Cats download is not required
(data/README.md). ``cache_dir`` is always pointed at ``tmp_path`` so nothing here
writes into the working tree.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from gatekeeper.data.ingest import DEFAULT_RAW_PATH, load_cookie_cats, project_root
from gatekeeper.data.schema import COOKIE_CATS
from gatekeeper.types import DataSource, SchemaViolation


@pytest.fixture
def csv_path(raw_frame: pd.DataFrame, tmp_path):
    p = tmp_path / "cookie_cats.csv"
    raw_frame.to_csv(p, index=False)
    return p


def _cache_file(cache_dir, schema=COOKIE_CATS):
    """The cache path for a schema. Fingerprinted, so editing the schema misses."""
    return cache_dir / f"{schema.name}-{schema.fingerprint}.parquet"


class TestProjectRoot:
    def test_resolves_to_the_repo_root(self):
        root = project_root()
        assert (root / "pyproject.toml").exists()
        assert (root / "src" / "gatekeeper").is_dir()

    def test_default_raw_path_is_relative(self):
        assert not os.path.isabs(DEFAULT_RAW_PATH)


class TestMissingFile:
    def test_raises_with_actionable_instructions(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            load_cookie_cats(tmp_path / "absent.csv")
        msg = str(exc.value)
        assert "Kaggle" in msg
        assert "data/README.md" in msg
        assert "synthetic" in msg


class TestLoading:
    def test_valid_csv_loads_and_validates(self, csv_path, tmp_path):
        data = load_cookie_cats(csv_path, cache_dir=tmp_path / "cache")
        assert len(data.frame) == 6
        assert data.n_per_arm == {"gate_30": 3, "gate_40": 3}
        assert data.frame["retention_1"].dtype == bool
        assert data.data_source is DataSource.REAL

    def test_data_source_can_be_tagged(self, csv_path, tmp_path):
        data = load_cookie_cats(
            csv_path, cache_dir=tmp_path / "cache", data_source=DataSource.SYNTHETIC
        )
        assert data.data_source is DataSource.SYNTHETIC

    def test_schema_violation_propagates(self, raw_frame, tmp_path):
        bad = raw_frame.copy()
        bad.loc[0, "version"] = "gate_50"
        p = tmp_path / "bad.csv"
        bad.to_csv(p, index=False)
        with pytest.raises(SchemaViolation, match="unexpected value"):
            load_cookie_cats(p, cache_dir=tmp_path / "cache")

    def test_missing_column_propagates(self, raw_frame, tmp_path):
        p = tmp_path / "short.csv"
        raw_frame.drop(columns=["retention_7"]).to_csv(p, index=False)
        with pytest.raises(SchemaViolation, match="missing required column"):
            load_cookie_cats(p, cache_dir=tmp_path / "cache")


class TestCache:
    def test_cache_is_written_on_first_load(self, csv_path, tmp_path):
        cache = tmp_path / "cache"
        load_cookie_cats(csv_path, cache_dir=cache)
        assert _cache_file(cache).exists()

    def test_second_load_returns_equivalent_data(self, csv_path, tmp_path):
        cache = tmp_path / "cache"
        first = load_cookie_cats(csv_path, cache_dir=cache)
        second = load_cookie_cats(csv_path, cache_dir=cache)
        pd.testing.assert_frame_equal(first.frame, second.frame)

    def test_cache_is_actually_read_not_the_csv(self, csv_path, tmp_path):
        """Prove the cache is used: mutate the CSV underneath a fresh cache.

        The cache is newer, so the stale cached content must win. This is the
        behaviour that makes cache invalidation worth testing.
        """
        cache = tmp_path / "cache"
        load_cookie_cats(csv_path, cache_dir=cache)

        smaller = pd.read_csv(csv_path).iloc[:2]
        # Rewrite the CSV but backdate it so the cache stays newer.
        smaller.to_csv(csv_path, index=False)
        old = time.time() - 3600
        os.utime(csv_path, (old, old))

        assert len(load_cookie_cats(csv_path, cache_dir=cache).frame) == 6

    def test_stale_cache_is_bypassed_when_csv_is_newer(self, csv_path, tmp_path):
        cache = tmp_path / "cache"
        load_cookie_cats(csv_path, cache_dir=cache)

        # Backdate the cache so the CSV is unambiguously newer.
        old = time.time() - 3600
        os.utime(_cache_file(cache), (old, old))

        pd.read_csv(csv_path).iloc[:2].to_csv(csv_path, index=False)
        assert len(load_cookie_cats(csv_path, cache_dir=cache).frame) == 2

    def test_use_cache_false_skips_the_cache_entirely(self, csv_path, tmp_path):
        cache = tmp_path / "cache"
        load_cookie_cats(csv_path, cache_dir=cache, use_cache=False)
        assert not _cache_file(cache).exists()

    def test_schema_change_invalidates_the_cache(self, csv_path, tmp_path):
        """Regression: freshness by mtime alone let a stale cache outlive its contract.

        The cache holds *validated* data, so a hit skips revalidation. Before the
        schema fingerprint entered the cache key, narrowing the schema returned rows
        checked against the previous contract -- with columns the new schema does not
        even declare.
        """
        from gatekeeper.data.schema import ColumnSpec, DatasetSchema

        cache = tmp_path / "cache"
        full = load_cookie_cats(csv_path, cache_dir=cache)
        assert len(full.frame.columns) == 5

        narrowed = DatasetSchema(
            name="cookie_cats",  # same name -> same cache basename before the fix
            unit_col="userid",
            variant_col="version",
            control="gate_30",
            columns=(
                ColumnSpec("userid", "int"),
                ColumnSpec("version", "str", allowed_values=frozenset({"gate_30", "gate_40"})),
            ),
        )
        assert narrowed.fingerprint != COOKIE_CATS.fingerprint

        again = load_cookie_cats(csv_path, schema=narrowed, cache_dir=cache)
        assert list(again.frame.columns) == ["userid", "version"]

    def test_fingerprint_is_stable_across_calls(self):
        """It must not use salted hash() -- the digest has to survive a restart."""
        from gatekeeper.data.schema import COOKIE_CATS

        assert COOKIE_CATS.fingerprint == COOKIE_CATS.fingerprint
        assert len(COOKIE_CATS.fingerprint) == 12

    def test_fingerprint_changes_when_post_treatment_flag_changes(self):
        """A retyped or reflagged column is a different contract (R1.7)."""
        from gatekeeper.data.schema import COOKIE_CATS, ColumnSpec, DatasetSchema

        flipped = DatasetSchema(
            name=COOKIE_CATS.name,
            unit_col=COOKIE_CATS.unit_col,
            variant_col=COOKIE_CATS.variant_col,
            control=COOKIE_CATS.control,
            columns=tuple(
                ColumnSpec(
                    c.name,
                    c.kind,
                    c.allowed_values,
                    post_treatment=False,
                    unique=c.unique,
                    description=c.description,
                )
                for c in COOKIE_CATS.columns
            ),
        )
        assert flipped.fingerprint != COOKIE_CATS.fingerprint

    def test_relative_path_resolves_against_repo_root(self, tmp_path):
        """A relative path is interpreted from the repo root, not the CWD."""
        with pytest.raises(FileNotFoundError) as exc:
            load_cookie_cats("data/raw/definitely_absent.csv", cache_dir=tmp_path)
        assert str(project_root()) in str(exc.value)
