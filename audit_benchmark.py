"""Re-score saved benchmark SQL offline, without Ollama or modifying original results.

Run from the project root:
    python audit_benchmark.py \\
        --base reports/base_original_qwen3_raw.jsonl \\
        --tuned reports/tuned_qwen_raw.jsonl

Output: reports/scorer_audit.json and reports/scorer_audit_cases.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from benchmark_audit_scoring import UNORDERED_CASE_IDS, has_rank_ties, tied_group_top
from db_engine import execute_query
from evaluate_baseline import load_cases, load_datasets, results_equal, case_fingerprint

ROOT = Path(__file__).resolve().parent


def load_saved(path: Path, cases_by_id: dict, raw_csv: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved model results: {path}")
    records = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        cid = r.get("id")
        if cid not in cases_by_id or cid in records:
            raise ValueError(f"Unknown or repeated benchmark ID {cid!r} in {path}:{lineno}")
        expected_fingerprint = case_fingerprint(cases_by_id[cid], raw_csv[cases_by_id[cid]["dataset"]])
        if r.get("case_fingerprint") != expected_fingerprint:
            raise ValueError(f"Benchmark/data mismatch for {cid} in {path}; audit stopped")
        if r.get("mode") != "raw":
            raise ValueError(f"Expected raw inference mode in {path}:{lineno}; found {r.get('mode')}")
        records[cid] = r
    if set(records) != set(cases_by_id):
        missing = sorted(set(cases_by_id) - set(records))
        raise ValueError(f"{path} contains {len(records)} / {len(cases_by_id)} cases; missing: {missing}")
    return records


def reference_all_ranked(case: dict, frame):
    # The 14 known group_top reference queries all have a terminal LIMIT.
    gold = case["gold_sql"]
    without_limit, n = re.subn(r"\s+LIMIT\s+\d+\s*;?\s*$", "", gold, flags=re.IGNORECASE)
    if n != 1:
        raise ValueError(f"Cannot read gold ranking LIMIT for {case['id']}")
    _, full = execute_query(frame, without_limit)
    k = int(re.search(r"\bLIMIT\s+(\d+)\s*$", gold, re.IGNORECASE).group(1))
    # First ORDER BY expression is always the projected aggregate score.
    order = gold.upper().split("ORDER BY", 1)[1].rsplit("LIMIT", 1)[0]
    match = re.search(r"\b(ASC|DESC)\b", order)
    asc = not match or match.group(1) == "ASC"
    return full, k, asc


def audit_record(case: dict, rec: dict, frame, gold, all_ranked):
    strict = bool(rec["final_correct"])
    adjusted_order = bool(case["ordered"]) and case["id"] not in UNORDERED_CASE_IDS
    def score(sql):
        if not sql:
            return False, "missing_sql"
        try:
            _, actual = execute_query(frame, sql)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {str(exc)[:140]}"
        if case["skill"] == "group_top":
            full, k, asc = all_ranked
            return tied_group_top(actual, full, k, ascending=asc), None
        return results_equal(actual, gold, ordered=adjusted_order), None
    first, first_error = score(rec.get("first_sql"))
    if rec.get("repaired"):
        final, final_error = score(rec.get("final_sql"))
    else:
        final, final_error = first, first_error
    # Guard against environment/fixture drift while allowing historically
    # unexecutable SQL to remain an ordinary incorrect benchmark response.
    if (rec.get("first_executed") and first_error) or (strict and final_error):
        raise RuntimeError(f"Previously successful SQL no longer executes for {case['id']}: "
                           f"first={first_error}, final={final_error}")
    reasons = []
    if case["id"] in UNORDERED_CASE_IDS and case["ordered"]:
        reasons.append("unrequested_order_removed")
    if all_ranked:
        full, k, _ = all_ranked
        if has_rank_ties(full, k):
            reasons.append("tied_ranking")
    if final != strict:
        reasons.append("recorded_score_changed")
    return {"old": strict, "new": final, "old_first": bool(rec["first_correct"]),
            "new_first": first, "executed": bool(rec["first_executed"]),
            "sql": rec.get("final_sql") or rec.get("first_sql"),
            "flags": reasons}


def score_summary(audits: dict) -> dict:
    return {
        "cases": len(audits),
        "original_correct": sum(x["old"] for x in audits.values()),
        "audited_correct": sum(x["new"] for x in audits.values()),
        "original_accuracy_pct": round(100 * sum(x["old"] for x in audits.values()) / len(audits), 2),
        "audited_accuracy_pct": round(100 * sum(x["new"] for x in audits.values()) / len(audits), 2),
        "changed_ids": [cid for cid, v in audits.items() if v["old"] != v["new"]],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "reports/base_original_qwen3_raw.jsonl")
    parser.add_argument("--tuned", type=Path, default=ROOT / "reports/tuned_qwen_raw.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/scorer_audit.json")
    args = parser.parse_args(argv)
    cases = load_cases()
    by_id = {c["id"]: c for c in cases}
    datasets = load_datasets(cases)
    raw_csv = {name: (ROOT / "benchmark/datasets" / f"{name}.csv").read_bytes() for name in datasets}
    base = load_saved(args.base, by_id, raw_csv)
    tuned = load_saved(args.tuned, by_id, raw_csv)
    if len({r["model"] for r in base.values()}) != 1 or len({r["model"] for r in tuned.values()}) != 1:
        raise ValueError("Model tag changed within a saved report")
    if next(iter(base.values()))["model"] != "qwen3:4b-q4_K_M":
        raise ValueError("Expected the corrected, original Qwen3 4B baseline report")
    if next(iter(tuned.values()))["model"] != "qwen-sql-analyst:v1":
        raise ValueError("Expected the fine-tuned Qwen SQL Analyst v1 report")
    base_audit, tuned_audit, cases_out = {}, {}, []
    for case in cases:
        cid = case["id"]
        frame = datasets[case["dataset"]].frame
        _, gold = execute_query(frame, case["gold_sql"])
        all_ranked = reference_all_ranked(case, frame) if case["skill"] == "group_top" else None
        a = audit_record(case, base[cid], frame, gold, all_ranked)
        b = audit_record(case, tuned[cid], frame, gold, all_ranked)
        base_audit[cid], tuned_audit[cid] = a, b
        def normal_sql(value):
            return re.sub(r"\s+", " ", (value or "").strip().rstrip(";")).lower()
        same_sql = normal_sql(a["sql"]) == normal_sql(b["sql"])
        cases_out.append({
            "id": cid, "question": case["question"], "difficulty": case["difficulty"],
            "skill": case["skill"], "base_old": a["old"], "base_audited": a["new"],
            "tuned_old": b["old"], "tuned_audited": b["new"],
            "same_sql": same_sql, "same_sql_differently_scored": same_sql and a["old"] != b["old"],
            "flags": sorted(set(a["flags"] + b["flags"])),
            "base_sql": a["sql"], "tuned_sql": b["sql"],
        })
    result = {
        "policy": "Offline audit only; original logs, reference SQL, datasets, and live evaluator unchanged. "
                  "Only genuinely unordered questions ignore row order; ranked group_top accepts tied maxima "
                  "at the cutoff with correct metrics, group membership, and primary ranking order. "
                  "Column count and six-decimal numeric comparison are unchanged.",
        "base_model": next(iter(base.values()))["model"],
        "tuned_model": next(iter(tuned.values()))["model"],
        "base": score_summary(base_audit),
        "tuned": score_summary(tuned_audit),
        "same_sql_differently_scored_ids": [r["id"] for r in cases_out if r["same_sql_differently_scored"]],
        "flagged_tied_rankings": [r["id"] for r in cases_out if "tied_ranking" in r["flags"]],
        "cases": cases_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = args.output.with_name(args.output.stem + "_cases.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cases_out[0]))
        writer.writeheader()
        for item in cases_out:
            writer.writerow({**item, "flags": ", ".join(item["flags"])})
    print("Saved:", args.output, "and", csv_path)
    for name in ("base", "tuned"):
        v = result[name]
        print(f"{name}: recorded {v['original_correct']}/80 ({v['original_accuracy_pct']}%), "
              f"audited {v['audited_correct']}/80 ({v['audited_accuracy_pct']}%), "
              f"changed cases: {v['changed_ids']}")
    print("Same SQL, different original score:", result["same_sql_differently_scored_ids"])
    print("Tied ranking cases:", result["flagged_tied_rankings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
