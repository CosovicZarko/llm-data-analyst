# Held-out Text-to-SQL Baseline Evaluation

80 hand-authored natural-language questions and matching DuckDB SQL queries across five single-table CSV schemas: retail transactions, support requests, staff shifts, SaaS invoices, and warehouse inventory. Each schema has 16 questions spanning simple filters, aggregates, grouping, top-k, dates, arithmetic, and conditional aggregates.

**Evaluation-only data:** Keep these five schemas and all 80 questions out of supervised fine-tuning examples. They are reserved for testing generalization. The reference SQL is never sent to the model; it is used to compute expected query results locally.

From the project root, with Ollama running and `qwen3:4b-instruct-2507-q4_K_M` installed:

```bash
python evaluate_baseline.py --validate-only
python evaluate_baseline.py --limit 10
python evaluate_baseline.py --limit 0
```

`--limit 10` samples two questions per dataset. `--limit 0` continues with all 80 and resumes previously recorded cases automatically. Each completed case is saved immediately to `reports/baseline_qwen.jsonl`, with a summary in `reports/baseline_qwen_summary.json`. To evaluate a different model, choose a different output path:

```bash
python evaluate_baseline.py --model another-ollama-model --output reports/another_model.jsonl --limit 0
```

**Metrics:** first SQL execution rate; first-attempt exact-result accuracy; exact-result accuracy after at most one DuckDB-error-guided repair; breakdowns by dataset, difficulty, and skill; per-question latency. A query that executes but returns the wrong values is scored incorrect and is not repaired. SQL guard rejections are not automatically repaired. Equivalent column aliases and insignificant numerical rounding are ignored; result shape must match and ranked-question order is checked. These are strict result-set comparisons, so some semantically valid queries that return extra/missing fields may be scored as incorrect.

The CSV tables are small, deterministic fixtures intended for model evaluation. Benchmark performance is not an estimate of all possible real-world questions or arbitrary uploaded spreadsheets.
