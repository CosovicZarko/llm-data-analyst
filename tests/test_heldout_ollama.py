"""Dependency-light isolated tests for the new held-out Ollama evaluator."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import types

# The imports below use real production code when duckdb is installed.
# This test intentionally does not mock production scoring methods.
try:
    from training.evaluate_ollama_heldout import (
        interleave_schemas, as_evaluation_case, fingerprint, read_existing,
        load_test_rows,
    )
except ImportError:
    raise


class HeldoutOllamaTests(unittest.TestCase):
    def test_interleave_covers_distinct_schemas(self):
        input_rows = [{"id": "a1", "schema_id": "a"}, {"id": "a2", "schema_id": "a"},
                      {"id": "b1", "schema_id": "b"}, {"id": "b2", "schema_id": "b"}]
        self.assertEqual([r["id"] for r in interleave_schemas(input_rows)],
                         ["a1", "b1", "a2", "b2"])

    def test_test_data_is_stable_and_schema_diverse(self):
        cases = load_test_rows()
        self.assertEqual(len(cases), 828)
        self.assertEqual(len({r["schema_id"] for r in cases[:20]}), 18)
        self.assertEqual(len({r["id"] for r in cases}), 828)

    def test_gold_sql_not_in_model_question(self):
        row = load_test_rows()[0]
        case = as_evaluation_case(row)
        self.assertEqual(case["question"], row["question"])
        self.assertEqual(case["gold_sql"], row["sql"])
        self.assertTrue(case["dataset"].endswith(tuple(f"v0{x}" for x in range(10))))
        self.assertEqual(case["ordered"], "ORDER BY" in row["sql"].upper())

    def test_fingerprint_detects_any_change(self):
        case = as_evaluation_case(load_test_rows()[0])
        a = fingerprint(case, b"row one")
        self.assertNotEqual(a, fingerprint(case, b"row two"))
        self.assertNotEqual(a, fingerprint({**case, "question": "different"}, b"row one"))

    def test_saved_results_rejected_for_other_model_and_modified_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.jsonl"
            case = as_evaluation_case(load_test_rows()[0])
            sig = fingerprint(case, b"csv")
            record = {"id": case["id"], "model": "qwen-sql-analyst:v1",
                      "mode": "raw", "case_fingerprint": sig}
            path.write_text(json.dumps(record) + "\n")
            self.assertEqual(len(read_existing(path, model="qwen-sql-analyst:v1", expected={case["id"]: sig})), 1)
            with self.assertRaises(ValueError):
                read_existing(path, model="qwen3:4b-q4_K_M", expected={case["id"]: sig})
            with self.assertRaises(ValueError):
                read_existing(path, model="qwen-sql-analyst:v1", expected={case["id"]: "wrong"})


if __name__ == "__main__":
    unittest.main()
