import unittest
try:
    from sql_guard import validate_sql
except ImportError:
    validate_sql = None

@unittest.skipIf(validate_sql is None, 'Install sqlglot to test SQL guard')
class GuardTests(unittest.TestCase):
    def test_aggregate_query(self):
        query = validate_sql('SELECT region, SUM(revenue) AS total FROM dataset GROUP BY region ORDER BY total DESC LIMIT 1')
        self.assertIn('SUM', query)

    def test_and_is_logical_operator_not_function(self):
        sql = validate_sql(
            "SELECT COUNT(*) FROM dataset WHERE subscription_plan = 'Enterprise' "
            "AND payment_state = 'Paid'"
        )
        self.assertIn(' AND ', sql)

    def test_or_is_logical_operator_not_function(self):
        sql = validate_sql(
            "SELECT COUNT(*) FROM dataset WHERE urgency = 'High' OR urgency = 'Critical'"
        )
        self.assertIn(' OR ', sql)

    def test_case_when_conditional_aggregation(self):
        sql = validate_sql(
            "SELECT ROUND(100.0 * SUM(CASE WHEN payment_state = 'Paid' "
            "THEN 1 ELSE 0 END) / COUNT(*), 2) FROM dataset"
        )
        self.assertIn('CASE WHEN', sql)

    def test_unknown_function_remains_blocked(self):
        with self.assertRaises(ValueError):
            validate_sql('SELECT RANDOM() FROM dataset')

    def test_rejects_write(self):
        for sql in ['DROP TABLE dataset', 'DELETE FROM dataset', 'COPY dataset TO "/tmp/x.csv"']:
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                validate_sql(sql)

    def test_rejects_multiple_statements(self):
        with self.assertRaises(ValueError):
            validate_sql('SELECT 1 FROM dataset; DROP TABLE dataset')

    def test_rejects_file_functions(self):
        with self.assertRaises(ValueError):
            validate_sql("SELECT * FROM read_csv('/etc/passwd')")

    def test_rejects_other_table(self):
        with self.assertRaises(ValueError):
            validate_sql('SELECT * FROM secrets')

    def test_rejects_join(self):
        with self.assertRaises(ValueError):
            validate_sql('SELECT * FROM dataset JOIN dataset b ON dataset.order_id = b.order_id')
