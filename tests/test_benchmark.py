"""Benchmark shape and scoring tests. DB-specific tests run when deps exist."""
import json
import csv
import sqlite3
from pathlib import Path
import unittest
from collections import Counter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASES = [json.loads(line) for line in (ROOT / 'benchmark' / 'cases.jsonl').read_text().splitlines() if line.strip()]


class ManifestTests(unittest.TestCase):
    def test_80_unique_heldout_questions(self):
        self.assertEqual(len(CASES), 80)
        self.assertEqual(len({c['id'] for c in CASES}), 80)
        self.assertEqual(len({(c['dataset'], c['question']) for c in CASES}), 80)

    def test_five_distinct_schemas(self):
        counts = Counter(c['dataset'] for c in CASES)
        self.assertEqual(len(counts), 5)
        self.assertTrue(all(n == 16 for n in counts.values()))
        for ds in counts:
            self.assertTrue((ROOT / 'benchmark' / 'datasets' / f'{ds}.csv').is_file())

    def test_gold_sql_executes_against_all_fixtures_in_sqlite(self):
        # Fast standard-library cross-check. The project-wide --validate-only
        # command additionally checks the same references under DuckDB.
        for name in sorted({case['dataset'] for case in CASES}):
            with (ROOT / 'benchmark' / 'datasets' / f'{name}.csv').open(newline='', encoding='utf8') as stream:
                data = list(csv.DictReader(stream))
            columns = list(data[0])
            with sqlite3.connect(':memory:') as connection:
                connection.execute('CREATE TABLE dataset (' + ', '.join('"'+c+'"' for c in columns) + ')')
                connection.executemany('INSERT INTO dataset VALUES (' + ','.join('?' for _ in columns) + ')',
                                       [[record[c] or None for c in columns] for record in data])
                for case in (c for c in CASES if c['dataset'] == name):
                    try:
                        rows = connection.execute(case['gold_sql']).fetchall()
                    except sqlite3.Error as exc:
                        self.fail(f"{case['id']} has invalid reference SQL: {exc}")
                    self.assertGreater(len(rows), 0, case['id'])
                    self.assertLessEqual(len(rows), 100, case['id'])

    def test_case_metadata(self):
        for case in CASES:
            self.assertIn(case['difficulty'], {'easy', 'medium', 'hard'})
            self.assertTrue(case['question'].strip())
            self.assertTrue(case['gold_sql'].startswith('SELECT '))
            self.assertIsInstance(case['ordered'], bool)


try:
    from evaluate_baseline import results_equal, load_cases, summarize
    import duckdb
    from dataset_utils import load_file
    from db_engine import execute_query
    from evaluate_baseline import evaluate_case
    HAVE_DB = True
except ImportError:
    HAVE_DB = False


@unittest.skipUnless(HAVE_DB, 'Install duckdb and sqlglot for DuckDB evaluation tests')
class EvaluationTests(unittest.TestCase):
    def test_reference_sql_runs_on_all_80_cases(self):
        cache = {}
        for case in CASES:
            if case['dataset'] not in cache:
                path = ROOT / 'benchmark' / 'datasets' / (case['dataset']+'.csv')
                cache[case['dataset']] = load_file(path.name, path.read_bytes())
            sql, frame = execute_query(cache[case['dataset']].frame, case['gold_sql'])
            self.assertIsInstance(frame, pd.DataFrame)
            self.assertTrue(len(frame) <= 100)

    def test_column_alias_ignored_and_order_ignored_by_default(self):
        gold = pd.DataFrame({'name': ['A', 'B'], 'total': [4, 2]})
        actual = pd.DataFrame({'s': [2, 4], 'n': ['B', 'A']})
        self.assertTrue(results_equal(actual, gold))
        self.assertFalse(results_equal(actual, gold, ordered=True))

    def test_wrong_result_is_failure_even_when_sql_executes(self):
        case = next(c for c in CASES if c['dataset'] == 'retail_transactions' and c['skill'] == 'count')
        path = ROOT / 'benchmark' / 'datasets' / 'retail_transactions.csv'
        data = load_file(path.name, path.read_bytes())
        rec = evaluate_case(case, data, model='mock', url='mock',
                            generate=lambda *args, **kwargs: 'SELECT 999 FROM dataset LIMIT 1',
                            repair=lambda *args, **kwargs: self.fail('Should not repair valid-but-wrong SQL'))
        self.assertTrue(rec['first_executed'])
        self.assertFalse(rec['first_correct'])
        self.assertFalse(rec['final_correct'])
        self.assertFalse(rec['repaired'])

    def test_duckdb_binder_error_triggers_one_repair(self):
        case = next(c for c in CASES if c['dataset'] == 'retail_transactions' and c['skill'] == 'group_top')
        path = ROOT / 'benchmark' / 'datasets' / 'retail_transactions.csv'
        data = load_file(path.name, path.read_bytes())
        calls = []
        def fix(q, schema, previous, err, **kw):
            calls.append(err)
            return case['gold_sql']
        rec = evaluate_case(case, data, model='mock', url='mock',
                            generate=lambda *args, **kwargs: 'SELECT item_name FROM dataset ORDER BY SUM(net_sales) DESC LIMIT 1',
                            repair=fix)
        self.assertEqual(len(calls), 1)
        self.assertTrue(rec['repaired'])
        self.assertFalse(rec['first_correct'])
        self.assertTrue(rec['final_correct'])

    def test_summary_counts_only_completed_cases(self):
        records = [dict(id='a',difficulty='easy',dataset='orders',skill='count',first_correct=True,
                        final_correct=True,first_executed=True,repaired=False,seconds=1),
                   dict(id='b',difficulty='hard',dataset='orders',skill='group',first_correct=False,
                        final_correct=True,first_executed=False,repaired=True,seconds=3)]
        summary = summarize(records, total=80)
        self.assertEqual(summary['evaluated'], 2)
        self.assertEqual(summary['first_pass_execution_accuracy_pct'], 50)
        self.assertEqual(summary['after_one_repair_execution_accuracy_pct'], 100)
        self.assertEqual(summary['repair_attempts'], 1)
        self.assertEqual(summary['recovered_from_first_failure'], 1)
