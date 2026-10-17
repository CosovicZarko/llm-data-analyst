"""Protect native Qwen3 thinking separation and strict SQL JSON parsing."""
import unittest
from unittest.mock import patch

import llm_client


class StubResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class ChatThinkJsonTests(unittest.TestCase):
    def test_reasoning_is_ignored_and_only_sql_is_returned(self):
        data = {"message": {
            "thinking": "Okay, let's think for 500 words about counting rows",
            "content": '{"sql":"SELECT COUNT(*) FROM dataset;"}',
        }, "done_reason": "stop"}
        with patch.object(llm_client.requests, "post", return_value=StubResponse(data)) as post:
            answer = llm_client.generate_sql("How many purchases?", "receipt_id: INT", model="qwen3:4b", mode="chat-think-json")
        self.assertEqual(answer, "SELECT COUNT(*) FROM dataset;")
        self.assertTrue(post.call_args.args[0].endswith('/api/chat'))
        args = post.call_args.kwargs
        self.assertTrue(args['json']['think'])
        self.assertEqual(args['json']['format']['required'], ['sql'])
        self.assertEqual(args['json']['model'], 'qwen3:4b')
        self.assertIn('receipt_id: INT', args['json']['messages'][-1]['content'])
        self.assertTrue(args['json']['messages'][-1]['content'].endswith('/no_think'))
        self.assertEqual(args['json']['options']['temperature'], 0)
        self.assertNotIn('gold_sql', str(args['json']))

    def test_raw_natural_language_cannot_be_scored(self):
        with self.assertRaises(llm_client.ModelOutputFormatError):
            llm_client._parse_structured_sql("Okay, let's see how many records exist")

    def test_thinking_in_sql_field_is_rejected(self):
        with self.assertRaises(llm_client.ModelOutputFormatError):
            llm_client._parse_structured_sql('{"sql":"Okay, let us think carefully"}')
        with self.assertRaises(llm_client.ModelOutputFormatError):
            llm_client._parse_structured_sql('{"sql":"<think>reasoning</think> SELECT 1"}')

    def test_missing_or_extra_fields_are_rejected(self):
        for response in ('{}', '{"sql":null}', '{"sql":"SELECT 1","thought":"reasoning"}'):
            with self.subTest(response=response), self.assertRaises(llm_client.ModelOutputFormatError):
                llm_client._parse_structured_sql(response)

    def test_truncated_result_not_scored(self):
        with self.assertRaises(llm_client.ModelOutputFormatError):
            llm_client._parse_structured_sql('{"sql":"SELECT COUNT(*) FROM dataset;"}', finish_reason='length')

    def test_empty_sql_json_not_scored(self):
        with self.assertRaises(llm_client.ModelOutputFormatError):
            llm_client._parse_structured_sql('{"sql":""}')

    def test_empty_answer_despite_thinking_not_scored(self):
        with patch.object(llm_client.requests, "post", return_value=StubResponse({
            'message': {'thinking': 'lots of reasoning', 'content': ''}
        })):
            with self.assertRaises(llm_client.ModelOutputFormatError):
                llm_client.generate_sql('count', 'schema', mode='chat-think-json')

    def test_other_inference_modes_unchanged(self):
        with patch.object(llm_client.requests, "post", return_value=StubResponse({
            'message': {'content': 'SELECT COUNT(*) FROM dataset;'}
        })) as post:
            result = llm_client.generate_sql('count', 'schema', mode='chat')
        self.assertEqual(result, 'SELECT COUNT(*) FROM dataset;')
        self.assertNotIn('format', post.call_args.kwargs['json'])
        self.assertNotIn('think', post.call_args.kwargs['json'])

    def test_repair_uses_same_structured_contract(self):
        with patch.object(llm_client.requests, 'post', return_value=StubResponse({
            'message': {'thinking': '', 'content': '{"sql":"SELECT COUNT(*) FROM dataset;"}'},
            'done_reason': 'stop'
        })) as post:
            sql = llm_client.repair_sql('count', 'schema', 'SELECT COUNT(bad) FROM dataset', 'Binder error', mode='chat-think-json')
        self.assertEqual(sql, 'SELECT COUNT(*) FROM dataset;')
        self.assertEqual(post.call_args.kwargs['json']['messages'][-2]['content'], 'SELECT COUNT(bad) FROM dataset')


if __name__ == '__main__':
    unittest.main()
