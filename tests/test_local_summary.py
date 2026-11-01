"""局域网摘要的缓存身份、未知请求保护及保守预算检查。"""

import json
from pathlib import Path
import tempfile
import unittest


class LocalSummaryTest(unittest.TestCase):
    def test_nonempty_length_limited_summary_matches_original_pytrio_behavior(self):
        from opensearch_vl.local_summary import summary_text, LocalSummaryError

        body = {"model": "model", "choices": [{"finish_reason": "length", "message": {"content": "partial summary"}}]}
        self.assertEqual(summary_text(body, "model"), "partial summary")
        body["choices"][0]["finish_reason"] = "stop"
        self.assertEqual(summary_text(body, "model"), "partial summary")
        body["choices"][0]["message"]["content"] = ""
        with self.assertRaises(LocalSummaryError):
            summary_text(body, "model")
        body["choices"][0]["message"]["content"] = "partial summary"
        body["choices"][0]["finish_reason"] = "tool_calls"
        with self.assertRaises(LocalSummaryError):
            summary_text(body, "model")
        body["choices"][0]["finish_reason"] = "stop"
        with self.assertRaises(LocalSummaryError):
            summary_text(body, "different-model")

    def test_different_deployments_never_share_summary_cache_identity(self):
        from opensearch_vl.local_summary import cache_identity

        first = {"model": "model", "base_url": "http://one/v1", "parameters": {"temperature": 0.3}}
        second = {**first, "base_url": "http://two/v1"}
        self.assertNotEqual(cache_identity(first), cache_identity(second))
        self.assertNotEqual(cache_identity(first), "Qwen/Qwen3.8-27B")
        self.assertEqual(cache_identity(first), cache_identity(dict(first)))

    def test_unknown_request_blocks_post_and_cannot_be_treated_as_zero_cost(self):
        from opensearch_vl.local_summary import LocalSummaryClient, LocalSummaryError
        from opensearch_vl.agent_tools import BudgetExhausted

        root = Path(__file__).resolve().parent / "outputs"
        root.mkdir(exist_ok=True)
        config = {"backend": "local-openai", "model": "model", "base_url": "http://127.0.0.1:1/v1",
                  "request_base_url": "http://127.0.0.1:1/v1", "parameters": {"max_tokens": 1024}}
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory)
            client = LocalSummaryClient(path, config, "not-used-offline", 32768)
            pending = client.pending_path("prompt")
            pending.write_text('{"state":"submitted"}')
            with self.assertRaises(LocalSummaryError):
                client.complete("prompt", "sample")
            self.assertFalse(client.ledger.exists())
            self.assertTrue(pending.exists())
            pending.unlink()
            client.ledger.write_text(json.dumps({"reserved_tokens": 32768}) + "\n")
            with self.assertRaises(BudgetExhausted):
                client.complete("prompt", "sample")
            self.assertEqual(len(client.ledger.read_text().splitlines()), 1)
