"""不发网络请求，核验原始轨迹存储、策略隔离及作者算法数值。"""

import ast
from collections import defaultdict
import io
from pathlib import Path
import unittest

import numpy as np
import pytrio as trio
from PIL import Image
import torch


class RLAdapterTest(unittest.TestCase):
    def test_update_count_uses_operation_type_not_filename_substring(self):
        from train_rl import count_optimizer_updates

        self.assertEqual(count_optimizer_updates([
            {"operation": "update"}, {"operation": "live-after-update"},
            {"operation": "save-state"}]), 1)

    def test_actor_recomputation_preserves_sampling_logprobs_separately(self):
        from train_rl import actor_metadata_from_response

        meta = {"start": 1, "old": [-0.7, -0.8], "advantage": 0.3, "normalizer": 1}
        response = {"loss_fn_outputs": [{"logprobs": {"data": [0., -0.71, -0.82]}}]}
        result = actor_metadata_from_response(response, meta)
        self.assertEqual(result["old"], [-0.71, -0.82])
        self.assertEqual(result["sampling_old"], [-0.7, -0.8])
        self.assertEqual(meta["old"], [-0.7, -0.8])
        with self.assertRaises(ValueError):
            actor_metadata_from_response({"loss_fn_outputs": [{"logprobs": {"data": [0.]}}]}, meta)

    def test_actor_recomputation_is_before_every_backward_and_optimizer(self):
        path = Path(__file__).resolve().parent / "train_rl.py"
        tree = ast.parse(path.read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                    and node.test.func.id == "any")
        operations = []
        for statement in loop.body:
            names = [n.args[1].value for n in ast.walk(statement) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "remote_operation"
                     and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)]
            operations.extend(names)
        self.assertLess(operations.index("old-actor"), operations.index("backward"))
        self.assertLess(operations.index("backward"), operations.index("update"))

    def test_checkpoint_verification_requires_matching_backends_and_real_change(self):
        from verify_rl_smoke import compare_logprobs

        result = compare_logprobs([-1., -2.], [-1.01, -2.02], [-1.015, -2.025])
        self.assertTrue(result["weight_effect_observed"])
        self.assertTrue(result["state_sampler_match"])
        self.assertFalse(compare_logprobs([-1.], [-1.], [-1.])["weight_effect_observed"])
        self.assertFalse(compare_logprobs([-1.], [-1.01], [-2.])["state_sampler_match"])
        with self.assertRaises(ValueError):
            compare_logprobs([-1.], [-1., -2.], [-1.])

    def test_smoke_is_limited_to_two_tasks_and_one_batch(self):
        import train_rl

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
        from rl_rollout import dump_prompt, load_prompt

        stream = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(stream, format="PNG")
        prompt = trio.ModelInput(chunks=[trio.types.EncodedTextChunk(tokens=[1, 2]),
            trio.ImageChunk(data=stream.getvalue(), format="png", expected_tokens=3)])
        restored = load_prompt(json.loads(json.dumps(dump_prompt(prompt))))
        self.assertEqual(restored, prompt)

    def test_rloo_matches_actual_author_function(self):
        from rl_prepare import rloo_advantages

        path = Path(__file__).resolve().parent.parent / "opensearch-vl/RL/rllm/verl/verl/trainer/ppo/core_algos.py"
        if not path.exists():
            path = Path(__file__).resolve().parent.parent / "OpenSearch-VL/RL/rllm/verl/verl/trainer/ppo/core_algos.py"
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
        from train_rl import group_training_data

        with self.assertRaises(ValueError):
            group_training_data([{"policy_id": p} for p in ["a", "a", "a", "b"]], [{}] * 4,
                                "seq-mean-token-sum")

    def test_empty_fatal_trajectories_still_have_group_statistics_but_no_datums(self):
        from train_rl import group_training_data

        records = [{"policy_id": "a", "trace": [], "termination": "context_limit"} for _ in range(4)]
        scores = [{"fatal_start": None, "reward": 0.} for _ in range(4)]
        self.assertEqual(group_training_data(records, scores, "seq-mean-token-sum"), ([], [], 0))
