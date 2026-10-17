"""UI backend routing tests: no Streamlit, Ollama, or network required."""
import unittest
from unittest.mock import Mock

from app_workflow import MODEL_OPTIONS, INFERENCE_MODE, execute_for_model, generate_for_model


class AppWorkflowTests(unittest.TestCase):
    def test_model_choices_are_installed_tags(self):
        self.assertEqual(INFERENCE_MODE, "raw")
        self.assertEqual(MODEL_OPTIONS["Fine-tuned Qwen3-4B (Google Cloud)"],
                         "qwen-sql-analyst:v1")
        self.assertEqual(MODEL_OPTIONS["Pretrained Qwen3-4B (Q4_K_M)"],
                         "qwen3:4b-q4_K_M")

    def test_both_models_generate_with_benchmark_raw_mode(self):
        for model in MODEL_OPTIONS.values():
            with self.subTest(model=model):
                generate = Mock(return_value="SELECT COUNT(*) FROM dataset")
                sql = generate_for_model("How many?", "Table: dataset", model,
                                         generator=generate)
                self.assertEqual(sql, "SELECT COUNT(*) FROM dataset")
                generate.assert_called_once_with("How many?", "Table: dataset",
                                                 model=model, mode="raw")

    def test_repair_receives_same_model_and_raw_mode(self):
        for model in MODEL_OPTIONS.values():
            with self.subTest(model=model):
                repair = Mock(return_value="SELECT COUNT(*) FROM dataset")

                def fake_runner(frame, question, schema, generated_sql, *, repairer):
                    return repairer(question, schema, generated_sql, "DuckDB Binder Error")

                result = execute_for_model(None, "How many?", "Table: dataset",
                                           "SELECT WRONG FROM dataset", model,
                                           runner=fake_runner, repairer=repair)
                self.assertEqual(result, "SELECT COUNT(*) FROM dataset")
                repair.assert_called_once_with(
                    "How many?", "Table: dataset", "SELECT WRONG FROM dataset",
                    "DuckDB Binder Error", model=model, mode="raw"
                )

    def test_executor_routes_sql_through_existing_runner(self):
        runner = Mock(return_value="executed")
        repair = Mock()
        result = execute_for_model("frame", "question", "schema", "SELECT 1",
                                   "qwen-sql-analyst:v1", runner=runner,
                                   repairer=repair)
        self.assertEqual(result, "executed")
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(runner.call_args.args,
                         ("frame", "question", "schema", "SELECT 1"))
        repair.assert_not_called()  # only invoked if the underlying runner needs it

    def test_unknown_model_is_rejected(self):
        generate = Mock()
        runner = Mock()
        with self.assertRaises(ValueError):
            generate_for_model("question", "schema", "unknown", generator=generate)
        with self.assertRaises(ValueError):
            execute_for_model(None, "q", "s", "SELECT 1", "unknown",
                              runner=runner)
        generate.assert_not_called()
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
