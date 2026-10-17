"""Coverage and leakage safeguards for the SQL training package."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import unittest

from training.build_dataset import DOMAINS, TRAIN_DOMAINS, VAL_DOMAINS, TEST_DOMAINS, _examples, _render_schema_and_csv
from training.train_qlora import _read, validate_splits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'training' / 'data'


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = _read(DATA / 'train.jsonl')
        cls.val = _read(DATA / 'validation.jsonl')
        cls.test = _read(DATA / 'test.jsonl')

    def test_split_sizes(self):
        self.assertEqual((len(self.train), len(self.val), len(self.test)), (2484, 828, 828))

    def test_holdout_domains_are_disjoint(self):
        partitions = [set(TRAIN_DOMAINS), set(VAL_DOMAINS), set(TEST_DOMAINS)]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])
        validate_splits(self.train, self.val)

    def test_unique_ids_and_schema_fingerprints(self):
        rows = self.train + self.val + self.test
        self.assertEqual(len({r['id'] for r in rows}), len(rows))
        self.assertEqual(len({r['schema_id'] for r in rows}), 90)

    def test_all_completion_sql_is_gold_sql(self):
        for row in self.train + self.val + self.test:
            self.assertEqual(row['completion'][0]['content'], row['sql'])
            self.assertEqual(row['prompt'][0]['role'], 'system')
            self.assertIn('Table: dataset', row['prompt'][1]['content'])
            self.assertIn('User question:', row['prompt'][1]['content'])
            self.assertTrue(row['sql'].startswith('SELECT '))

    def test_training_has_priority_skills(self):
        skills = Counter(row['skill'] for row in self.train)
        for skill in ('having', 'group_top', 'arithmetic', 'conditional', 'no_implicit_filter', 'filter_having'):
            self.assertGreaterEqual(skills[skill], 100, msg=skill)

    def test_seeded_fixture_is_deterministic(self):
        a = _render_schema_and_csv('cinema_bookings', 0, 20261015)
        b = _render_schema_and_csv('cinema_bookings', 0, 20261015)
        self.assertEqual(a, b)

    def test_all_sources_are_synthetic_not_benchmark_fixtures(self):
        for row in self.train + self.val + self.test:
            self.assertNotIn(row['schema_id'].rsplit('_v', 1)[0],
                             {'retail_transactions', 'saas_invoices', 'staff_shifts',
                              'support_requests', 'warehouse_inventory'})

    def test_manifest_hashes_match(self):
        import hashlib
        manifest = json.loads((DATA / 'manifest.json').read_text())
        for split in ('train','validation','test'):
            actual = hashlib.sha256((DATA / f'{split}.jsonl').read_bytes()).hexdigest()
            self.assertEqual(actual, manifest['sha256'][split])

    def test_every_fixture_has_eight_columns(self):
        for f in (DATA / 'fixtures').glob('*.csv'):
            header = f.read_text(encoding='utf-8').splitlines()[0].strip().split(',')
            self.assertEqual(len(set(header)), 8)


if __name__ == '__main__':
    unittest.main()
