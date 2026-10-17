"""Conservative validation of single-table, read-only DuckDB SELECT queries.

This is an MVP defense-in-depth guard, not an OS process sandbox.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

ALLOWED_FUNCTIONS = {
    "SUM", "COUNT", "AVG", "MIN", "MAX", "ROUND", "ABS", "LOWER", "UPPER",
    "TRIM", "LENGTH", "COALESCE", "NULLIF", "YEAR", "MONTH", "DAY",
    "DATE_TRUNC", "CAST", "TRY_CAST", "EXTRACT", "DATE_PART",
    "CASE", "IF",
}


def validate_sql(text: str) -> str:
    # A SQLGlot parse tree is required; never execute unchecked model text.
    try:
        statements = sqlglot.parse(text, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise ValueError("The LLM produced invalid SQL syntax.") from exc
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Exactly one SQL query is allowed.")
    root = statements[0]
    if not isinstance(root, exp.Select):
        raise ValueError("Only SELECT queries are supported.")
    if sum(isinstance(node, exp.Select) for node in root.walk()) != 1:
        raise ValueError("Subqueries are not supported in version 1.")
    blocked_types = tuple(t for t in (
        getattr(exp, "Join", None), getattr(exp, "With", None),
        getattr(exp, "Union", None), getattr(exp, "Intersect", None),
        getattr(exp, "Except", None), getattr(exp, "Lateral", None),
        getattr(exp, "Command", None), getattr(exp, "Into", None),
        getattr(exp, "Parameter", None),
    ) if t is not None)
    if any(isinstance(node, blocked_types) for node in root.walk()):
        raise ValueError("This SQL feature is not supported for uploaded tables.")
    tables = list(root.find_all(exp.Table))
    if len(tables) != 1 or tables[0].name.lower() != "dataset":
        raise ValueError("Queries may read only the uploaded table named dataset.")
    if tables[0].args.get("db") or tables[0].args.get("catalog"):
        raise ValueError("Qualified database access is not supported.")
    for node in root.walk():
        # SQLGlot represents AND/OR/XOR as Connector expressions that
        # also inherit Func. They are SQL operators, not callable functions.
        if isinstance(node, exp.Func) and not isinstance(node, exp.Connector):
            name = (node.sql_name() or "").upper()
            if name not in ALLOWED_FUNCTIONS:
                raise ValueError(f"SQL function {name or 'unknown'} is not permitted.")
        if isinstance(node, exp.Table) and node.name.lower() != "dataset":
            raise ValueError("Unknown table.")
    # Canonical SQL is used as returned by the guard, not the original text.
    return root.sql(dialect="duckdb")
