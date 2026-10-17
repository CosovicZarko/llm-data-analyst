import unittest
from unittest.mock import patch
from llm_client import extract_sql, generate_sql, repair_sql


class TestClient(unittest.TestCase):
    def test_fenced_sql(self):
        self.assertEqual(extract_sql('```sql\nSELECT * FROM dataset;\n```'), 'SELECT * FROM dataset;')

    def test_raw_sql(self):
        self.assertEqual(extract_sql('SQL: SELECT COUNT(*) FROM dataset'), 'SELECT COUNT(*) FROM dataset')

    @patch('llm_client.requests.post')
    def test_request_contract(self, post):
        post.return_value.json.return_value = {'message': {'content': 'SELECT COUNT(*) FROM dataset'}}
        result = generate_sql('Count rows', 'Table: dataset', model='test')
        self.assertEqual(result, 'SELECT COUNT(*) FROM dataset')
        self.assertFalse(post.call_args.kwargs['json']['stream'])

    @patch('llm_client.requests.post')
    def test_repair_receives_schema_failed_sql_and_error(self, post):
        post.return_value.json.return_value = {'message': {'content': 'SELECT product, SUM(revenue) AS total FROM dataset GROUP BY product'}}
        output = repair_sql('Best-selling product?', 'product, revenue',
                            'SELECT product FROM dataset ORDER BY SUM(revenue) DESC',
                            'Binder Error: product must appear in GROUP BY', model='test')
        self.assertIn('GROUP BY product', output)
        messages = post.call_args.kwargs['json']['messages']
        self.assertEqual(messages[2]['role'], 'assistant')
        self.assertIn('GROUP BY', messages[3]['content'])
        self.assertIn('product, revenue', messages[1]['content'])
