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
