import unittest
from dataset_utils import load_file, normalize_columns
import pandas as pd


class DatasetTests(unittest.TestCase):
    def test_sample_csv(self):
        with open('data/sample_sales.csv', 'rb') as f:
            d = load_file('sales.csv', f.read())
        self.assertEqual(len(d.frame), 10)
        self.assertEqual(d.frame['revenue'].sum(), 7450)
        self.assertIn('Table: dataset', d.schema)

    def test_columns_are_safe_and_unique(self):
        d = normalize_columns(pd.DataFrame([[1, 2, 3]], columns=['Order Total', 'Order-Total', '2024 Sales']))
        self.assertEqual(list(d.frame.columns), ['order_total', 'order_total_2', 'col_2024_sales'])

    def test_refuses_other_file_types(self):
        with self.assertRaises(ValueError):
            load_file('x.txt', b'one,two\n1,2')

    def test_size_limit(self):
        with self.assertRaises(ValueError):
            load_file('x.csv', b'x'*(8 * 1024 * 1024 + 1))
