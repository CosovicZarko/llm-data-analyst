"""Execution-accuracy benchmark for base or tuned text-to-SQL models (Ollama).

80 reserved evaluation questions across 5 novel single-table schemas. Reference
SQL is used exclusively to compute scores and is never included in model input.
Results are appended after each question and may be resumed safely.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
import hashlib
import itertools
import json
import math
from numbers import Number
from pathlib import Path
import time
from typing import Any, Callable

import duckdb
import pandas as pd

from dataset_utils import load_file
from db_engine import execute_query
from llm_client import MODEL, URL, ModelOutputFormatError, generate_sql, repair_sql

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "benchmark"
DEFAULT_OUTPUT = ROOT / "reports" / "baseline_qwen.jsonl"


def load_cases() -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in (BENCHMARK / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [c["id"] for c in cases]
    if len(cases) != 80 or len(set(ids)) != 80:
        raise ValueError("Expected 80 uniquely identified evaluation questions")
    # Interleave schemas to ensure that a 5- or 10-question smoke run covers
    # the different domains rather than only the first dataset.
    groups = defaultdict(list)
    for case in cases:
        groups[case["dataset"]].append(case)
    return [case for index in range(max(map(len, groups.values())))
            for dataset in sorted(groups)
            for case in groups[dataset][index:index + 1]]


def load_datasets(cases: list[dict]) -> dict[str, Any]:
    result = {}
    for name in sorted({c["dataset"] for c in cases}):
        path = BENCHMARK / "datasets" / (name + ".csv")
        result[name] = load_file(path.name, path.read_bytes())
    return result


def _cell(value: Any) -> tuple[str, Any]:
    if pd.isna(value):
        return ("null", "")
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return ("time", value.isoformat())
    if isinstance(value, Number) and not isinstance(value, bool):
        val = float(value)
        if math.isnan(val):
            return ("null", "")
        if not math.isfinite(val):
            return ("nonfinite", str(val))
        # Ignore insignificant floating-point differences (e.g. AVG outputs).
        return ("num", round(val, 6))
    if isinstance(value, Decimal):
        return ("num", round(float(value), 6))
    if isinstance(value, bool):
        return ("bool", value)
    return ("text", str(value))


def results_equal(actual: pd.DataFrame, expected: pd.DataFrame, *, ordered: bool = False) -> bool:
    """Compare executed result tables, ignoring SQL column aliases.

    Row order matters for ranked questions; it is ignored otherwise. Column
    order can vary, but the same number of output fields is required.
    """
    if actual.shape != expected.shape:
        return False
    ncols = actual.shape[1]
    if ncols > 6:  # avoid factorial work on arbitrary model SELECT * outputs
        return False
    expected_rows = [tuple(_cell(v) for v in row) for row in expected.itertuples(index=False, name=None)]
    actual_rows = [tuple(_cell(v) for v in row) for row in actual.itertuples(index=False, name=None)]
    right = expected_rows if ordered else Counter(expected_rows)
    for permutation in itertools.permutations(range(ncols)):
        permuted = [tuple(row[i] for i in permutation) for row in actual_rows]
        left = permuted if ordered else Counter(permuted)
        if left == right:
            return True
    return False


def case_fingerprint(case: dict, dataset_csv_bytes: bytes) -> str:
    return hashlib.sha256(
        json.dumps(case, sort_keys=True).encode("utf-8") + b"\n" + dataset_csv_bytes
    ).hexdigest()


def evaluate_case(case: dict, dataset: Any, *, model: str, url: str, mode: str = "chat",
                  generate: Callable = generate_sql,
                  repair: Callable = repair_sql,
                  execute: Callable = execute_query) -> dict[str, Any]:
    """Score one case using the same SQL guard and repair policy as the app."""
    start = time.perf_counter()
    output = {"id": case["id"], "dataset": case["dataset"],
              "difficulty": case["difficulty"], "skill": case["skill"],
              "question": case["question"], "model": model, "mode": mode,
              "first_sql": None, "final_sql": None, "first_executed": False,
              "first_correct": False, "final_correct": False,
              "repaired": False, "error": None, "seconds": None}
    try:
        _, expected = execute(dataset.frame, case["gold_sql"])
        generated = generate(case["question"], dataset.schema,
                             model=model, url=url, mode=mode)
        output["first_sql"] = generated
        try:
            first_sql, first_rows = execute(dataset.frame, generated)
            output["first_executed"] = True
            output["final_sql"] = first_sql
            correct = results_equal(first_rows, expected, ordered=case["ordered"])
            output["first_correct"] = correct
            output["final_correct"] = correct
        except duckdb.Error as first_error:
            # Only errors from DuckDB execution trigger repair. SQL guard
            # rejections are never sent for an automatic bypass attempt.
            output["repaired"] = True
            fixed = repair(case["question"], dataset.schema, generated,
                           str(first_error), model=model, url=url, mode=mode)
            output["final_sql"] = fixed
            final_sql, final_rows = execute(dataset.frame, fixed)
            output["final_sql"] = final_sql
            output["final_correct"] = results_equal(
                final_rows, expected, ordered=case["ordered"])
    except ModelOutputFormatError as exc:
        # Do not count a prompt/template breakdown as SQL model accuracy.
        output["inference_format_error"] = True
        output["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    except Exception as exc:
        output["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    finally:
        output["seconds"] = round(time.perf_counter() - start, 3)
    return output


def summarize(records: list[dict], *, total: int) -> dict[str, Any]:
    count = len(records)
    def rate(n: int, d: int) -> float | None:
        return round(100 * n / d, 2) if d else None
    first_correct = sum(bool(r["first_correct"]) for r in records)
    final_correct = sum(bool(r["final_correct"]) for r in records)
    executable = sum(bool(r["first_executed"]) for r in records)
    repair_attempts = sum(bool(r["repaired"]) for r in records)
    details = {}
    for group_key in ("difficulty", "dataset", "skill"):
        details[group_key] = {}
        for key in sorted({r[group_key] for r in records}):
            subset = [r for r in records if r[group_key] == key]
            details[group_key][key] = {
                "evaluated": len(subset),
                "first_pass_accuracy_pct": rate(sum(bool(r["first_correct"]) for r in subset), len(subset)),
                "final_accuracy_pct": rate(sum(bool(r["final_correct"]) for r in subset), len(subset)),
            }
    return {
        "evaluated": count, "total_questions": total,
        "first_execution_success_pct": rate(executable, count),
        "first_pass_execution_accuracy_pct": rate(first_correct, count),
        "after_one_repair_execution_accuracy_pct": rate(final_correct, count),
        "repair_attempts": repair_attempts,
        "recovered_from_first_failure": sum(bool(r["repaired"] and r["final_correct"]) for r in records),
        "average_seconds_per_question": round(sum(r["seconds"] for r in records) / count, 2) if count else None,
        "by_group": details,
        "note": "Strict result-table equivalence; column aliases ignored, row order checked for ranking tasks. Correct SQL with extra/missing columns may be scored incorrect. Reference SQL never appears in the model prompt.",
    }


def _print_summary(summary: dict) -> None:
    print(f"\nCompleted: {summary['evaluated']}/{summary['total_questions']}")
    print(f"First-pass result accuracy: {summary['first_pass_execution_accuracy_pct']}%")
    print(f"After error repair:          {summary['after_one_repair_execution_accuracy_pct']}%")
    print(f"First-query executes:        {summary['first_execution_success_pct']}%")
    print(f"Repair attempts:             {summary['repair_attempts']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="First N interleaved questions; 0 runs all 80 (default: 10)")
    parser.add_argument("--model", default=MODEL, help="Ollama model tag")
    parser.add_argument("--url", default=URL,
                        help="Ollama URL (usually /api/chat; raw mode switches it to /api/generate)")
    parser.add_argument("--mode", choices=("chat", "chat-nothink", "raw", "raw-select", "chat-think-json"), default="chat",
                        help="Ollama chat; chat-nothink; raw Qwen3; raw-select; or chat-think-json (structured SQL with separated thinking)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Append-only JSONL results file")
    parser.add_argument("--validate-only", action="store_true", help="Execute gold SQL on all datasets; no Ollama calls")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    cases = load_cases()
    datasets = load_datasets(cases)
    raw_data = {name: (BENCHMARK / "datasets" / (name + ".csv")).read_bytes() for name in datasets}
    fingerprints = {c["id"]: case_fingerprint(c, raw_data[c["dataset"]]) for c in cases}
    print(f"Benchmark: {len(cases)} questions, {len(datasets)} unfamiliar schemas, model: {args.model}", flush=True)
    # Check every reference answer before contacting the model, so dataset
    # errors cannot be counted as model failures.
    for case in cases:
        try:
            execute_query(datasets[case["dataset"]].frame, case["gold_sql"])
        except Exception as exc:
            raise RuntimeError(f"Reference SQL invalid for {case['id']}: {exc}") from exc
    print("All 80 reference SQL queries executed successfully.", flush=True)
    if args.validate_only:
        return 0
    target = cases if args.limit == 0 else cases[:args.limit]
    output_path = args.output.expanduser()
    if args.mode != "chat" and output_path.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error("Use --output reports/<unique-name>.jsonl for alternate inference modes to protect the original baseline")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec["id"]
            if cid in existing:
                raise ValueError(f"Duplicate case in existing results: {cid}")
            if cid not in fingerprints or rec.get("case_fingerprint") != fingerprints[cid]:
                raise ValueError(f"Benchmark changed for {cid}; choose a new --output file.")
            if rec.get("model") != args.model:
                raise ValueError("Previous results used a different model; choose a new --output file.")
            # Old baseline records didn't record a mode; they used /api/chat.
            if rec.get("mode", "chat") != args.mode:
                raise ValueError("Previous results used a different inference mode; choose a new --output file.")
            existing[cid] = rec
    pending = [c for c in target if c["id"] not in existing]
    print(f"Selected {len(target)} questions; {len(existing)} saved, {len(pending)} pending.", flush=True)
    try:
        with output_path.open("a", encoding="utf-8") as out:
            for idx, case in enumerate(pending, 1):
                print(f"[{idx}/{len(pending)}] {case['id']} ({case['difficulty']}) {case['question']}", flush=True)
                rec = evaluate_case(case, datasets[case["dataset"]],
                                    model=args.model, url=args.url, mode=args.mode)
                if rec.get("inference_format_error"):
                    # A reasoning-only response indicates the inference setup
                    # still needs fixing. Stop without saving a misleading FAIL
                    # so this case can be retried with the same output filename.
                    print("  STOPPED (inference format failure; not scored): " + rec["error"], flush=True)
                    print("No result saved for this case. Check the model/template before continuing.", flush=True)
                    return 2
                rec["case_fingerprint"] = fingerprints[case["id"]]
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                existing[case["id"]] = rec
                label = "PASS" if rec["final_correct"] else "FAIL"
                print(f"  {label}  first={rec['first_correct']} repaired={rec['repaired']} time={rec['seconds']}s", flush=True)
                if rec["error"] and ("Cannot connect to Ollama" in rec["error"] or
                                       "Model request failed: 404" in rec["error"]):
                    print("Ollama connection/model unavailable; stopping after first failure.", flush=True)
                    break
    except KeyboardInterrupt:
        print("\nInterrupted. Completed questions remain saved; rerun to resume.", flush=True)
    scored = [existing[c["id"]] for c in cases if c["id"] in existing]
    summary = summarize(scored, total=len(cases))
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _print_summary(summary)
    print(f"Results: {output_path}\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
