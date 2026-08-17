"""Load raw experiment files into validated :class:`ExperimentData`.

Ingest validates and caches; it never repairs. A file that does not match its
schema raises :class:`~gatekeeper.types.SchemaViolation` (Rules §5).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gatekeeper.data.schema import COOKIE_CATS, DatasetSchema, ExperimentData
from gatekeeper.types import DataSource

__all__ = ["DEFAULT_RAW_PATH", "load_cookie_cats", "project_root"]


def project_root() -> Path:
    """Repository root, resolved from this file's location.

    ``src/gatekeeper/data/ingest.py`` -> up four levels.
    """
    return Path(__file__).resolve().parents[3]


DEFAULT_RAW_PATH = "data/raw/cookie_cats.csv"


def load_cookie_cats(
    path: str | Path | None = None,
    *,
    schema: DatasetSchema = COOKIE_CATS,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    data_source: DataSource = DataSource.REAL,
) -> ExperimentData:
    """Load and validate the Cookie Cats dataset.

    Parameters
    ----------
    path
        Path to the CSV. Defaults to ``data/raw/cookie_cats.csv`` under the repo
        root.
    schema
        Contract to validate against.
    use_cache
        If ``True``, read a Parquet cache when it is newer than the CSV, and write
        one after a successful load. The cache holds *validated* data, so a cache
        hit skips revalidation.
    cache_dir
        Where to keep the Parquet cache. Defaults to ``data/processed`` under the
        repo root; injectable so tests do not write into the working tree.
    data_source
        Provenance tag. Leave as ``REAL`` for the actual download.

    Returns
    -------
    ExperimentData
        Validated, correctly typed data.

    Raises
    ------
    FileNotFoundError
        If the CSV is absent, with instructions for obtaining it.
    SchemaViolation
        If the file does not match ``schema``.
    """
    root = project_root()
    csv_path = Path(path) if path is not None else root / DEFAULT_RAW_PATH
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    if not csv_path.exists():
        raise FileNotFoundError(
            f"dataset not found at {csv_path}\n"
            "Download cookie_cats.csv from the Kaggle 'Cookie Cats' / 'Mobile Games "
            "A/B Testing' dataset and place it there. See data/README.md.\n"
            "To build and test without it, use gatekeeper.data.synthetic."
        )

    cache_root = Path(cache_dir) if cache_dir is not None else root / "data" / "processed"
    cache_path = cache_root / f"{schema.name}.parquet"
    if use_cache and cache_path.exists() and cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
        frame = pd.read_parquet(cache_path)
        return ExperimentData(frame=frame, schema=schema, data_source=data_source)

    data = ExperimentData.from_frame(pd.read_csv(csv_path), schema=schema, data_source=data_source)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.frame.to_parquet(cache_path, index=False)

    return data
