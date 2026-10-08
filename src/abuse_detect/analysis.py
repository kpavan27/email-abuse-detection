"""Exploratory analysis in SQL (DuckDB) over the processed parquet file."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


def list_queries() -> list[str]:
    return sorted(
        p.name for p in resources.files("abuse_detect.sql").iterdir() if p.name.endswith(".sql")
    )


def run_query(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    sql = resources.files("abuse_detect.sql").joinpath(name).read_text()
    return con.execute(sql).df()


def run_all(parquet_path: str | Path, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    # DDL can't take bound parameters
    safe_path = str(parquet_path).replace("'", "''")
    con.execute(f"CREATE VIEW messages AS SELECT * FROM read_parquet('{safe_path}')")
    results = {}
    for name in list_queries():
        df = run_query(con, name)
        df.to_csv(out_dir / name.replace(".sql", ".csv"), index=False)
        log.info("%s -> %d rows", name, len(df))
        results[name] = df
    return results
