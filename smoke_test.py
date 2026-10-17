"""Check the real database path without running or downloading an LLM."""
from pathlib import Path

from dataset_utils import load_file
from db_engine import execute_query


def main() -> None:
    sample = Path(__file__).parent / "data" / "sample_sales.csv"
    dataset = load_file(sample.name, sample.read_bytes())
    sql, rows = execute_query(
        dataset.frame,
        "SELECT region, SUM(revenue) AS total FROM dataset "
        "GROUP BY region ORDER BY total DESC LIMIT 1",
    )
    assert len(rows) == 1 and rows.iloc[0]["region"] == "North"
    assert int(rows.iloc[0]["total"]) == 2300
    print("DuckDB SQL smoke test passed. Top region: North, revenue: 2300")
    print("SQL:", sql)


if __name__ == "__main__":
    main()
