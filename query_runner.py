"""One execution-guided repair attempt for DuckDB query errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import duckdb
import pandas as pd

from db_engine import execute_query
from llm_client import repair_sql


@dataclass(frozen=True)
class QueryExecution:
    sql: str
    rows: pd.DataFrame
    repaired: bool
    initial_error: str | None = None


def execute_with_one_repair(
    frame: pd.DataFrame,
    question: str,
    schema: str,
    generated_sql: str,
    *,
    executor: Callable = execute_query,
    repairer: Callable = repair_sql,
) -> QueryExecution:
    """Run generated SQL, retry once after DuckDB errors; validate each attempt.

    The executor always routes through the existing SQLGlot security guard.
    SQL guard rejections are not automatically repaired or bypassed.
    """
    try:
        sql, rows = executor(frame, generated_sql)
        return QueryExecution(sql=sql, rows=rows, repaired=False)
    except duckdb.Error as exc:
        error_message = str(exc)

    corrected_sql = repairer(question, schema, generated_sql, error_message)
    # If the second attempt fails, propagate the error instead of inventing data.
    sql, rows = executor(frame, corrected_sql)
    return QueryExecution(sql=sql, rows=rows, repaired=True, initial_error=error_message)
