import unittest
from unittest.mock import Mock

try:
    import duckdb
    import pandas as pd
    from query_runner import execute_with_one_repair
except ImportError:
    duckdb = None


@unittest.skipIf(duckdb is None, "Install duckdb and sqlglot for execution tests")
class RepairTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"product": ["Laptop"], "revenue": [1200]})
        self.bad = "SELECT product FROM dataset ORDER BY SUM(revenue) DESC LIMIT 1"
        self.good = "SELECT product, SUM(revenue) AS total FROM dataset GROUP BY product ORDER BY total DESC LIMIT 1"

    def test_binder_error_triggers_one_repair(self):
        executor = Mock(side_effect=[
            duckdb.BinderException("column product must appear in GROUP BY clause"),
            (self.good, pd.DataFrame({"product": ["Laptop"], "total": [1200]})),
        ])
        repairer = Mock(return_value=self.good)
        result = execute_with_one_repair(self.frame, "Best-selling product?", "Table: dataset", self.bad,
                                         executor=executor, repairer=repairer)
        self.assertTrue(result.repaired)
        self.assertEqual(result.rows.iloc[0]["product"], "Laptop")
        self.assertEqual(executor.call_count, 2)
        repairer.assert_called_once()
        self.assertIn("GROUP BY", repairer.call_args.args[3])

    def test_valid_sql_does_not_repair(self):
        executor = Mock(return_value=(self.good, pd.DataFrame({"total": [1200]})))
        repairer = Mock()
        result = execute_with_one_repair(self.frame, "Question", "schema", self.good,
                                         executor=executor, repairer=repairer)
        self.assertFalse(result.repaired)
        repairer.assert_not_called()
        executor.assert_called_once()

    def test_second_sql_failure_is_not_retried_again(self):
        executor = Mock(side_effect=[
            duckdb.BinderException("bad aggregation"),
            duckdb.BinderException("still bad"),
        ])
        repairer = Mock(return_value=self.bad)
        with self.assertRaises(duckdb.BinderException):
            execute_with_one_repair(self.frame, "Question", "schema", self.bad,
                                    executor=executor, repairer=repairer)
        self.assertEqual(executor.call_count, 2)
        repairer.assert_called_once()

    def test_validation_errors_do_not_trigger_repair(self):
        executor = Mock(side_effect=ValueError("SQL function READ_CSV is not permitted"))
        repairer = Mock()
        with self.assertRaises(ValueError):
            execute_with_one_repair(self.frame, "Question", "schema", self.bad,
                                    executor=executor, repairer=repairer)
        repairer.assert_not_called()
