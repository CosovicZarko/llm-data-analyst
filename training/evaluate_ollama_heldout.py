"""Evaluate an installed Ollama model on the domain-held-out synthetic SQL test.

This uses the same SQL validator, DuckDB execution, and bounded one-repair
scoring as evaluate_baseline.py. No prompts contain the gold SQL. Existing
benchmark outputs are not read or modified. Scores use the *original strict*
result-table comparator, not the separately audited 80-question overrides.

Examples:
    python -m training.evaluate_ollama_heldout --limit 20
    python -m training.evaluate_ollama_heldout --limit 0  # resume to all 828
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dataset_utils import load_file
from evaluate_baseline import evaluate_case, summarize
from llm_client import URL
from db_engine import execute_query

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "training" / "data"
DEFAULT_MODEL = "qwen-sql-analyst:v1"
DEFAULT_OUTPUT = ROOT / "reports" / "heldout_test_tuned_raw.jsonl"


def interleave_schemas(records: list[dict]) -> list[dict]:
    """Make a small smoke run cover many of the 18 held-out schemas."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["schema_id"]].append(row)
    return [row for index in range(max(map(len, grouped.values()), default=0))
            for schema in sorted(grouped)
            for row in grouped[schema][index:index+1]]


def load_test_rows(path: Path = TEST_DATA / "test.jsonl") -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError("Held-out test dataset is empty")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate test identifiers")
    return interleave_schemas(rows)


def as_evaluation_case(row: dict) -> dict[str, Any]:
    return {
        "id": row["id"], "dataset": row["schema_id"],
        "difficulty": "synthetic", "skill": row["skill"],
        "question": row["question"], "gold_sql": row["sql"],
        # Match training.evaluate_synthetic's existing strict rubric.
        "ordered": "ORDER BY" in row["sql"].upper(),
    }


def fingerprint(case: dict, csv_bytes: bytes) -> str:
    return hashlib.sha256(
        json.dumps(case, sort_keys=True).encode("utf-8") + b"\n" + csv_bytes
    ).hexdigest()


def read_existing(path: Path, *, model: str, expected: dict[str, str]) -> dict[str, dict]:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        case_id = record["id"]
        if case_id in records:
            raise ValueError(f"Duplicate saved result for {case_id}")
        if case_id not in expected or record.get("case_fingerprint") != expected[case_id]:
            raise ValueError(f"Test data changed for {case_id}; choose a NEW output file")
        if record.get("model") != model or record.get("mode") != "raw":
            raise ValueError("Output already belongs to a different model or inference mode")
        records[case_id] = record
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Installed Ollama model tag")
    parser.add_argument("--limit", type=int, default=20, help="20 diverse questions by default; 0 = all 828")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Append-only result file, model-specific")
    parser.add_argument("--url", default=URL, help="Ollama local API URL")
    parser.add_argument("--validate-only", action="store_true", help="Check selected gold SQL without Ollama calls")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.model != DEFAULT_MODEL and args.output.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error("Provide --output with a distinct filename when changing --model")

    rows = load_test_rows()
    chosen_rows = rows if args.limit == 0 else rows[:args.limit]
    if args.limit > 0 and len(chosen_rows) < args.limit:
        parser.error(f"Only {len(rows)} held-out questions available")
    cases = [as_evaluation_case(row) for row in chosen_rows]

    fixtures: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for case in cases:
        schema_id = case["dataset"]
        if schema_id not in fixtures:
            file = TEST_DATA / "fixtures" / f"{schema_id}.csv"
            payload = file.read_bytes()
            fixtures[schema_id] = (load_file(file.name, payload), payload)
        data, payload = fixtures[schema_id]
        hashes[case["id"]] = fingerprint(case, payload)
        try:
            execute_query(data.frame, case["gold_sql"])
        except Exception as exc:
            raise RuntimeError(f"Reference SQL did not execute for {case['id']}: {exc}") from exc
    print(f"Validated gold SQL for {len(cases)} held-out questions "+
          f"across {len(fixtures)} unseen schemas.", flush=True)
    if args.validate_only:
        return 0

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = read_existing(output, model=args.model, expected=hashes)
    pending = [c for c in cases if c["id"] not in saved]
    print(f"Model: {args.model}; mode: raw; saved: {len(saved)}; pending: {len(pending)}", flush=True)
    with output.open("a", encoding="utf-8") as result_file:
        for index, case in enumerate(pending, 1):
            record = evaluate_case(
                case, fixtures[case["dataset"]][0], model=args.model,
                url=args.url, mode="raw",
            )
            if record.get("inference_format_error"):
                print(f"STOPPED: {case['id']} output format error (not scored): {record['error']}", flush=True)
                return 2
            if record.get("error") and (
                "Cannot connect to Ollama" in record["error"] or
                "Model request failed: 404" in record["error"]
            ):
                print(f"STOPPED: model unavailable for {case['id']} (not scored)", flush=True)
                return 2
            record["case_fingerprint"] = hashes[case["id"]]
            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            result_file.flush()
            saved[case["id"]] = record
            print(f"[{index}/{len(pending)}] {case['id']}: " +
                  ("PASS" if record["final_correct"] else "FAIL"), flush=True)

    ordered_results = [saved[c["id"]] for c in cases if c["id"] in saved]
    summary = summarize(ordered_results, total=len(rows))
    summary.update({
        "split": "test", "inference_backend": "Ollama", "inference_mode": "raw",
        "model_id": args.model,
        "evaluation_policy": "Strict execution-result tables; ORDER BY inferred from gold SQL.",
        "caveat": "Synthetic unseen domains share question templates with training; this is not a production-data test.",
    })
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Done: {summary['evaluated']}/{summary['total_questions']} questions, "
          f"accuracy {summary['after_one_repair_execution_accuracy_pct']}%", flush=True)
    print(f"Results: {output}\nSummary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
