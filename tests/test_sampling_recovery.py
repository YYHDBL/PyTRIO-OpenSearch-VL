"""显式拒绝请求保留预算预留；未知请求继续阻止重发。"""

import json
from pathlib import Path
import tempfile
import unittest

from pytrio.errors.actor import ActorAPIError


class SamplingRecoveryTest(unittest.TestCase):
    def test_rejected_request_is_not_unknown_and_does_not_release_reservation(self):
        from opensearch_vl.agent_baseline import TokenBudget
        from opensearch_vl.agent_tools import BudgetExhausted

        root = Path(__file__).resolve().parent / "outputs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "tokens.jsonl"
            budget = TokenBudget(path, 20)
            entry = budget.reserve_request("id", 4, 6)
            self.assertTrue(budget.has_unconfirmed("id"))
            budget.record_rejection(entry, ActorAPIError("rejected", status_code=400))
            restored = TokenBudget(path, 20)
            self.assertFalse(restored.has_unconfirmed("id"))
            self.assertNotIn("tokens", json.loads(path.read_text()))
            restored.reserve_request("id", 4, 6)
            with self.assertRaises(BudgetExhausted):
                restored.reserve_request("another", 1, 1)

    def test_timeout_and_server_error_cannot_be_declared_rejected(self):
        from opensearch_vl.agent_baseline import TokenBudget

        root = Path(__file__).resolve().parent / "outputs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "tokens.jsonl"
            budget = TokenBudget(path, 20)
            entry = budget.reserve_request("id", 4, 6)
            before = path.read_bytes()
            for status in [0, 500, 502]:
                with self.assertRaises(ValueError):
                    budget.record_rejection(entry, ActorAPIError("unknown", status_code=status))
                self.assertTrue(budget.has_unconfirmed("id"))
                self.assertEqual(path.read_bytes(), before)
