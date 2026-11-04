"""The one cache helper every fetch goes through (Week 2 Data Conventions, Caching)."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def cached(path: Path, fetch_fn: Callable[[], pd.DataFrame],
           validate_fn: Callable[[pd.DataFrame], None],
           force_refresh: bool = False, columns=None) -> pd.DataFrame:
    """Read `path` if it exists; otherwise fetch, validate, write and return.

    Only force_refresh=True refetches, or a file lacking one of `columns` (a cache
    from an older schema; a cache hit is not validated again). The cached table is
    validated on the way in, so a file on disk has always passed the checks.
    """
    path = Path(path)
    stale = columns is not None and path.exists() and not set(columns) <= set(pq.read_schema(path).names)
    if stale:
        log.info("cache from an older schema, rebuilding: %s", path.name)
    if path.exists() and not force_refresh and not stale:
        log.info("cache hit: %s", path.name)
        return pd.read_parquet(path)
    start = time.perf_counter()
    df = fetch_fn()
    validate_fn(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("fetched %s in %.1f s (%d rows)", path.name, time.perf_counter() - start, len(df))
    return df
