"""不发网络请求，核验原始轨迹存储、策略隔离及作者算法数值。"""

import ast
from collections import defaultdict
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pytrio as trio
from PIL import Image
import torch


class RLAdapterTest(unittest.TestCase):
    def test_account_quota_stops_instead_of_discarding_remaining_training_tasks(self):
        from opensearch_vl.rl_diagnostic import JudgeQuotaExhausted, raise_on_judge_quota
        from opensearch_vl.train_rl import collect_group

        raise_on_judge_quota({"error": {"code": "1302", "message": "普通请求限流"}})
        root = Path(__file__).resolve().parent / "outputs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory)
            with self.assertRaises(JudgeQuotaExhausted):
                collect_group("q", "policy", path,
                    lambda replica: raise_on_judge_quota({"error": {"code": "1308", "message": "已达到5小时使用上限"}}))
            self.assertFalse((path / "q.skipped.json").exists())

    def test_tool_budget_is_run_scoped_and_survives_resume_without_erasing_history(self):
        from opensearch_vl.agent_tools import ToolEnvironment, BudgetExhausted

        root = Path(__file__).resolve().parent / "outputs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory)
            cache = path / "cache"
            cache.mkdir()
            history = cache / "requests.jsonl"
            history.write_text('{"kind":"text_search","at":1}\n' * 2)
            original = history.read_bytes()
            run_ledger = path / "tool-requests.jsonl"
            tools = ToolEnvironment(cache, text_limit=2, image_limit=1, request_ledger=run_ledger)
            tools.charge("text_search")
            tools.charge("text_search")
            tools.charge("image_search")
            self.assertEqual(history.read_bytes(), original)
            with self.assertRaises(BudgetExhausted):
                tools.charge("text_search")
            restored = ToolEnvironment(cache, text_limit=2, image_limit=1, request_ledger=run_ledger)
            with self.assertRaises(BudgetExhausted):
                restored.charge("image_search")
            self.assertEqual(len(run_ledger.read_text().splitlines()), 3)
            self.assertEqual(ToolEnvironment(cache).ledger, history)

    def test_selected_complete_groups_do_not_require_failed_unselected_tasks(self):
        from opensearch_vl.train_rl import validate_diagnostic_selection

        diagnostic = {"complete": False, "groups": {
            "ready": {"complete": True, "scored": 4},
            "failed": {"complete": False, "scored": 3}}, "deferred": {"failed-r3": "HTTP 502"}}
        validate_diagnostic_selection(["ready"], diagnostic)
        with self.assertRaises(ValueError):
            validate_diagnostic_selection(["failed"], diagnostic)
        with self.assertRaises(ValueError):
            validate_diagnostic_selection(["missing"], diagnostic)
        diagnostic["deferred"]["ready-r0"] = "尚未确认"
        with self.assertRaises(ValueError):
            validate_diagnostic_selection(["ready"], diagnostic)

    def test_service_failure_skips_whole_group_and_preserves_pending(self):
        from opensearch_vl.rl_diagnostic import judge_request, parse_score
        from opensearch_vl.train_rl import collect_group

        root = Path(__file__).resolve().parent / "outputs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory)
            pending = path / "accuracy.pending.json"
            pending.write_text('{"sha256":"existing-unconfirmed-request"}')

            def collect(replica):
                return judge_request("unchanged", "accuracy", path, {}, parse_score)

            self.assertIsNone(collect_group("q", "policy", path, collect))
            skipped = json.loads((path / "q.skipped.json").read_text())
            self.assertEqual(skipped["completed_rollouts"], 0)
            self.assertEqual(skipped["policy_id"], "policy")
            self.assertTrue(pending.exists())
            self.assertFalse((path / "requests.jsonl").exists())
            pending.unlink()
            self.assertIsNone(collect_group("q", "policy", path, collect))
            with self.assertRaises(ValueError):
                collect_group("q", "different-policy", path, collect)

    def test_budget_and_programming_errors_are_not_skipped_as_service_failures(self):
        from opensearch_vl.agent_baseline import TokenBudget
        from opensearch_vl.agent_tools import BudgetExhausted
        from opensearch_vl.train_rl import collect_group

        root = Path(__file__).resolve().parent / "outputs"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory)
            budget = TokenBudget(path / "tokens.jsonl", 1)
            with self.assertRaises(BudgetExhausted):
                collect_group("q", "policy", path, lambda replica: budget.reserve_request("q", 2, 1))
            self.assertFalse((path / "q.skipped.json").exists())
            with self.assertRaises(ValueError):
                collect_group("q", "policy", path, lambda replica: int("invalid"))

    def test_update_count_uses_operation_type_not_filename_substring(self):
        from opensearch_vl.train_rl import count_optimizer_updates

        self.assertEqual(count_optimizer_updates([
            {"operation": "update"}, {"operation": "live-after-update"},
            {"operation": "save-state"}]), 1)

    def test_actor_recomputation_preserves_sampling_logprobs_separately(self):
        from opensearch_vl.train_rl import actor_metadata_from_response

        meta = {"start": 1, "old": [-0.7, -0.8], "advantage": 0.3, "normalizer": 1}
        response = {"loss_fn_outputs": [{"logprobs": {"data": [0., -0.71, -0.82]}}]}
        result = actor_metadata_from_response(response, meta)
        self.assertEqual(result["old"], [-0.71, -0.82])
        self.assertEqual(result["sampling_old"], [-0.7, -0.8])
        self.assertEqual(meta["old"], [-0.7, -0.8])
        with self.assertRaises(ValueError):
            actor_metadata_from_response({"loss_fn_outputs": [{"logprobs": {"data": [0.]}}]}, meta)

    def test_actor_recomputation_is_before_every_backward_and_optimizer(self):
        path = Path(__file__).resolve().parent.parent / "opensearch_vl/train_rl.py"
        tree = ast.parse(path.read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name) and node.test.id == "updated")
        operations = []
        for statement in loop.body:
            names = [n.args[1].value for n in ast.walk(statement) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "remote_operation"
                     and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)]
            operations.extend(names)
        self.assertLess(operations.index("old-actor"), operations.index("backward"))
        self.assertLess(operations.index("backward"), operations.index("update"))

    def test_checkpoint_verification_requires_matching_backends_and_real_change(self):
        from opensearch_vl.verify_rl_smoke import compare_logprobs

        result = compare_logprobs([-1., -2.], [-1.01, -2.02], [-1.015, -2.025])
        self.assertTrue(result["weight_effect_observed"])
        self.assertTrue(result["state_sampler_match"])
        self.assertFalse(compare_logprobs([-1.], [-1.], [-1.])["weight_effect_observed"])
        self.assertFalse(compare_logprobs([-1.], [-1.01], [-2.])["state_sampler_match"])
        with self.assertRaises(ValueError):
            compare_logprobs([-1.], [-1., -2.], [-1.])

    def test_smoke_is_limited_to_two_tasks_and_one_batch(self):
        from opensearch_vl import train_rl

        self.assertTrue(hasattr(train_rl, "validate_selection"))
        available = {"a", "b", "c", "d"}
        train_rl.validate_selection(["a", "b"], available, smoke=True)
        with self.assertRaises(ValueError):
            train_rl.validate_selection(["a", "b", "c"], available, smoke=True)
        with self.assertRaises(ValueError):
            train_rl.validate_selection(["a", "a"], available, smoke=True)
        with self.assertRaises(ValueError):
            train_rl.validate_selection(["a", "b"], available, smoke=False)

    def test_image_prompt_survives_json_roundtrip(self):
        import json
        from opensearch_vl.rl_rollout import dump_prompt, load_prompt

        stream = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(stream, format="PNG")
        prompt = trio.ModelInput(chunks=[trio.types.EncodedTextChunk(tokens=[1, 2]),
            trio.ImageChunk(data=stream.getvalue(), format="png", expected_tokens=3)])
        restored = load_prompt(json.loads(json.dumps(dump_prompt(prompt))))
        self.assertEqual(restored, prompt)

    def test_rloo_matches_actual_author_function(self):
        from opensearch_vl.rl_prepare import rloo_advantages

        path = Path(__file__).resolve().parents[2] / "opensearch-vl/RL/rllm/verl/verl/trainer/ppo/core_algos.py"
        if not path.exists():
            path = Path(__file__).resolve().parents[2] / "OpenSearch-VL/RL/rllm/verl/verl/trainer/ppo/core_algos.py"
        if not path.exists():
            self.skipTest("需要在本仓库同级放置作者 OpenSearch-VL 源码，才能比较作者函数")
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_rloo_outcome_advantage")
        function.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
        scope = {"torch": torch, "np": np, "defaultdict": defaultdict}
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        values = [0., 0.2, 0.6, 1.]
        expected, _ = scope["compute_rloo_outcome_advantage"](
            torch.tensor(values, dtype=torch.float64).reshape(4, 1), torch.ones(4, 1), np.array(["q"] * 4))
        np.testing.assert_allclose(rloo_advantages(values, [False] * 4), expected[:, 0].numpy())

    def test_training_group_rejects_mixed_policy_versions(self):
        from opensearch_vl.train_rl import group_training_data

        with self.assertRaises(ValueError):
            group_training_data([{"policy_id": p} for p in ["a", "a", "a", "b"]], [{}] * 4,
                                "seq-mean-token-sum")

    def test_empty_fatal_trajectories_still_have_group_statistics_but_no_datums(self):
        from opensearch_vl.train_rl import group_training_data

        records = [{"policy_id": "a", "trace": [], "termination": "context_limit"} for _ in range(4)]
        scores = [{"fatal_start": None, "reward": 0.} for _ in range(4)]
        self.assertEqual(group_training_data(records, scores, "seq-mean-token-sum"), ([], [], 0))
