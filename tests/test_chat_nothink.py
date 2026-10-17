"""Check request-level Ollama thinking toggle without loading any local models."""
import unittest
from unittest.mock import patch
import llm_client


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class ThinkingModesTest(unittest.TestCase):
    def test_chat_nothink_uses_native_chat_endpoint_and_disables_thinking(self):
        with patch.object(llm_client.requests, "post", return_value=DummyResponse({
            "message": {"content": "SELECT COUNT(*) FROM dataset;", "thinking": ""}
        })) as send:
            sql = llm_client.generate_sql("How many records?", "table dataset",
                                          model="qwen3:4b", mode="chat-nothink")
        self.assertEqual(sql, "SELECT COUNT(*) FROM dataset;")
        self.assertEqual(send.call_args.args[0], llm_client.URL)
        payload = send.call_args.kwargs["json"]
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertNotIn("prompt", payload)

    def test_original_chat_unchanged(self):
        with patch.object(llm_client.requests, "post", return_value=DummyResponse({
            "message": {"content": "SELECT 1;"}
        })) as send:
            self.assertEqual(llm_client.generate_sql("One?", "table dataset", mode="chat"), "SELECT 1;")
        self.assertNotIn("think", send.call_args.kwargs["json"])

    def test_raw_uses_generate_and_preserves_prefix(self):
        with patch.object(llm_client.requests, "post", return_value=DummyResponse({
            "response": "SELECT 2;"
        })) as send:
            self.assertEqual(llm_client.generate_sql("Two?", "table dataset", mode="raw"), "SELECT 2;")
        self.assertTrue(send.call_args.args[0].endswith("/api/generate"))
        payload = send.call_args.kwargs["json"]
        self.assertTrue(payload["raw"])
        self.assertIn("<think>\n\n</think>", payload["prompt"])

if __name__ == "__main__":
    unittest.main()
