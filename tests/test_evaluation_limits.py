"""评测生成预算与未完成请求的真实 token 预留。"""

import json
from pathlib import Path
import shutil
import unittest

import agent_baseline


class EvaluationLimitsTest(unittest.TestCase):
    def test_incomplete_action_is_not_executed_or_patched(self):
        text = '<tool_call>{"name":"text_search","arguments":{"q":"query"}}'
        self.assertIsNone(agent_baseline.parse_action(text))

    def test_32k_new_tokens_are_available_when_the_service_window_allows(self):
        self.assertTrue(hasattr(agent_baseline, "generation_budget"))
        self.assertEqual(agent_baseline.generation_budget(8000, 65536, 32768), 32768)
        self.assertEqual(agent_baseline.generation_budget(65520, 65536, 32768), 16)
        self.assertEqual(agent_baseline.generation_budget(65536, 65536, 32768), 0)

    def test_inflight_reservation_includes_full_prompt_and_requested_output(self):
        self.assertTrue(hasattr(agent_baseline.TokenBudget, "reserve_request"))
        root = Path("outputs/test-evaluation-reserve")
        self.assertFalse(root.exists())
        root.mkdir(parents=True)
        try:
            ledger = root / "tokens.jsonl"
            budget = agent_baseline.TokenBudget(ledger, 100000)
            entry = budget.reserve_request("length-probe", 32000, 32768)
            self.assertEqual(entry["reserved_tokens"], 64768)
            restored = agent_baseline.TokenBudget(ledger, 100000)
            with self.assertRaises(agent_baseline.BudgetExhausted):
                restored.reserve_request("next-request", 40000, 1000)
            self.assertEqual(len(ledger.read_text().splitlines()), 1)
        finally:
            shutil.rmtree(root)

    def test_legacy_unknown_usage_is_not_assumed_zero(self):
        self.assertTrue(hasattr(agent_baseline.TokenBudget, "reserve_request"))
        root = Path("outputs/test-evaluation-unknown")
        self.assertFalse(root.exists())
        root.mkdir(parents=True)
        try:
            ledger = root / "tokens.jsonl"
            ledger.write_text(json.dumps({"sample_id": "old-uncertain-request"}) + "\n")
            with self.assertRaises(ValueError):
                agent_baseline.TokenBudget(ledger, 100000).reserve_request("next", 100, 100)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
