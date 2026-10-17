"""Checks for example questions and multi-question input. No Streamlit required."""
import unittest

import pandas as pd

from example_questions import MAX_BATCH_QUESTIONS, prepare_questions, suggested_questions


class ExampleQuestionsTests(unittest.TestCase):
    def test_sample_sales_uses_real_columns(self):
        frame = pd.DataFrame({
            'order_id': [1, 2], 'region': ['West', 'East'],
            'product': ['Laptop', 'Monitor'], 'revenue': [25.0, 35.0],
        })
        examples = suggested_questions(frame)
        self.assertGreaterEqual(len(examples), 5)
        self.assertIn('Which region has the highest total revenue?', examples)
        self.assertIn('What is the total revenue?', examples)
        self.assertTrue(all('warehouse' not in q.lower() for q in examples))

    def test_prioritizes_region_in_original_sales_example(self):
        frame = pd.DataFrame({
            'order_id': [1, 2], 'customer': ['Acme', 'Beta'],
            'region': ['West', 'East'], 'product': ['Laptop', 'Monitor'],
            'revenue': [1200, 800],
        })
        self.assertEqual(
            suggested_questions(frame)[0],
            'Which region has the highest total revenue?',
        )

    def test_text_only_columns(self):
        frame = pd.DataFrame({'category': ['a', 'b'], 'status': ['yes', 'no']})
        examples = suggested_questions(frame)
        self.assertIn('How many records are in this dataset?', examples)
        self.assertFalse(any('total revenue' in q for q in examples))

    def test_numeric_only_dataset(self):
        frame = pd.DataFrame({'quantity': [2, 3], 'unit_price': [8, 9]})
        examples = suggested_questions(frame)
        self.assertIn('What is the total unit_price?', examples)
        self.assertNotIn('Which region has the highest total revenue?', examples)

    def test_does_not_use_identifier_as_metric(self):
        frame = pd.DataFrame({'order_id': [1, 2], 'customer': ['A', 'B']})
        examples = suggested_questions(frame)
        self.assertFalse(any('total order_id' in q for q in examples))

    def test_dedupe_and_custom(self):
        self.assertEqual(
            prepare_questions(['How many rows?', ' How many rows? '], 'Another question?'),
            ['How many rows?', 'Another question?'],
        )

    def test_empty_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Choose an example'):
            prepare_questions([], ' ')

    def test_maximum_batch_is_enforced(self):
        with self.assertRaisesRegex(ValueError, 'at most'):
            prepare_questions([str(i) for i in range(MAX_BATCH_QUESTIONS)], 'extra')

    def test_overlong_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '1,200'):
            prepare_questions([], 'x' * 1201)


if __name__ == '__main__':
    unittest.main()
