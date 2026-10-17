import unittest
from pathlib import Path
try:
    from db_engine import execute_query
except ImportError:
    execute_query = None

from dataset_utils import load_file

@unittest.skipIf(execute_query is None, 'Install duckdb and sqlglot for engine tests')
class ExecutionTests(unittest.TestCase):
    def test_sales_aggregation(self):
        data = load_file('data.csv', Path('data/sample_sales.csv').read_bytes())
        sql, result = execute_query(data.frame, 'SELECT region, SUM(revenue) AS total FROM dataset GROUP BY region ORDER BY total DESC LIMIT 1')
        self.assertEqual(result.iloc[0]['region'], 'North')
        self.assertEqual(int(result.iloc[0]['total']), 2300)
