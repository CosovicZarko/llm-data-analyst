"""Regression tests for model-completion-prefilled SQL-only raw inference."""
import unittest
from unittest.mock import patch

import llm_client


class DummyResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class RawSelectTests(unittest.TestCase):
    def test_raw_select_prefixes_select_and_no_think(self):
        with patch.object(llm_client.requests, 'post', return_value=DummyResponse({
            'response': 'COUNT(*) FROM dataset;'
        })) as post:
            output = llm_client.generate_sql('How many records?', 'Table: dataset',
                                             model='qwen3:4b', mode='raw-select')
        self.assertEqual(output, 'SELECT COUNT(*) FROM dataset;')
        url = post.call_args.args[0]
        payload = post.call_args.kwargs['json']
        self.assertTrue(url.endswith('/api/generate'))
        self.assertTrue(payload['raw'])
        self.assertEqual(payload['model'], 'qwen3:4b')
        self.assertTrue(payload['prompt'].endswith('<|im_start|>assistant\n<think>\n\n</think>\nSELECT '))
        self.assertIn('/no_think<|im_end|>', payload['prompt'])
        self.assertNotIn('gold_sql', payload['prompt'])

    def test_restarted_select_is_not_duplicated(self):
        self.assertEqual(llm_client._complete_prefilled_select('SELECT COUNT(*) FROM dataset;'),
                         'SELECT COUNT(*) FROM dataset;')

    def test_empty_response_is_inference_error(self):
        with self.assertRaises(llm_client.ModelOutputFormatError):
            llm_client._complete_prefilled_select(' ')

    def test_reasoning_response_is_not_mis_scored_as_sql(self):
        sample = "Okay, let's see. The user is asking how many purchases were recorded."
        with self.assertRaises(llm_client.ModelOutputFormatError):
            llm_client._complete_prefilled_select(sample)

    def test_reasoning_tag_is_not_accepted(self):
        with self.assertRaises(llm_client.ModelOutputFormatError):
            llm_client._complete_prefilled_select('<think>Counting rows</think> COUNT(*) FROM dataset;')

    def test_original_raw_mode_unchanged(self):
        with patch.object(llm_client.requests, 'post', return_value=DummyResponse({
            'response': 'SELECT COUNT(*) FROM dataset;'
        })) as post:
            result = llm_client.generate_sql('How many?', 'Table: dataset', mode='raw')
        self.assertEqual(result, 'SELECT COUNT(*) FROM dataset;')
        self.assertNotIn('/no_think', post.call_args.kwargs['json']['prompt'])
        self.assertFalse(post.call_args.kwargs['json']['prompt'].endswith('SELECT '))


if __name__ == '__main__':
    unittest.main()
