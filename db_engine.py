"""Create a fresh ephemeral DuckDB table for every request."""

from __future__ import annotations
import duckdb
import pandas as pd
from sql_guard import validate_sql

MAX_RESULT_ROWS = 100


def execute_query(frame: pd.DataFrame, model_sql: str) -> tuple[str, pd.DataFrame]:
    clean_sql = validate_sql(model_sql)
    # Use a separate in-memory database for each query. No uploaded files saved.
    conn = duckdb.connect(database=":memory:")
    try:
        conn.execute("SET threads = 2")
        conn.execute("SET memory_limit = '256MB'")
        conn.register("_incoming_frame", frame)
        conn.execute("CREATE TABLE dataset AS SELECT * FROM _incoming_frame")
        conn.unregister("_incoming_frame")
        # Disable external file/network access before executing model SQL.
        conn.execute("SET enable_external_access = false")
        conn.execute("SET autoinstall_known_extensions = false")
        conn.execute("SET autoload_known_extensions = false")
        conn.execute("SET lock_configuration = true")
        limited_sql = f"SELECT * FROM ({clean_sql}) AS _result LIMIT {MAX_RESULT_ROWS}"
        result = conn.execute(limited_sql).df()
        return clean_sql, result
    finally:
        conn.close()
