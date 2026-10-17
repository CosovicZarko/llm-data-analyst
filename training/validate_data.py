"""Run every synthetic reference SQL through the actual DuckDB + SQLGlot app.

Validates all train, validation and final-test reference queries without
contacting the LLM, Hugging Face, or an external database.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from dataset_utils import load_file
from db_engine import execute_query

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "data"


def main() -> None:
    cache = {}
    total = 0
    counts = Counter()
    for split in ("train", "validation", "test"):
        path = DATA / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing {path}; run python -m training.build_dataset")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            schema_id = row["schema_id"]
            if schema_id not in cache:
                data = (DATA / "fixtures" / f"{schema_id}.csv").read_bytes()
                cache[schema_id] = load_file(schema_id + ".csv", data)
            try:
                execute_query(cache[schema_id].frame, row["sql"])
            except Exception as exc:
                raise RuntimeError(f"{split} {row['id']}: {exc}") from exc
            counts[split] += 1
            total += 1
    print(f"DuckDB + SQLGlot reference validation passed: {total} queries "
          f"({', '.join(f'{k}={v}' for k, v in counts.items())})")


if __name__ == "__main__":
    main()
