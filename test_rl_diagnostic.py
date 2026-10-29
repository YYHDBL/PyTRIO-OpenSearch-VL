"""RL诊断的采样参数、分组和奖励边界。"""

import unittest


class DiagnosticTest(unittest.TestCase):
    def test_sampling_defaults_preserve_base_and_diagnostic_is_random(self):
        import agent_baseline

        self.assertTrue(hasattr(agent_baseline, "sampling_parameters"))
        base = agent_baseline.sampling_parameters({"generation_stop_tokens": ["<|im_end|>"]}, 100, {})
        self.assertEqual(base.temperature, 0)
        self.assertEqual(base.seed, 42)
        diag = agent_baseline.sampling_parameters({"generation_stop_tokens": ["<|im_end|>"],
            "temperature": 0.7, "top_p": 1.0}, 97, {"sampling_seed": 1007})
        self.assertEqual(diag.temperature, 0.7)
        self.assertEqual(diag.max_tokens, 97)
        self.assertEqual(diag.seed, 1007)

    def test_fatal_prefix_and_terminal_answer_are_not_benchmark_extraction(self):
        import importlib.util

        self.assertIsNotNone(importlib.util.find_spec("rl_diagnostic"))
        from rl_diagnostic import reward_parts

        good = {"assistant": '<think>x</think><tool_call>{}</tool_call>', "action": {"name": "text_search", "arguments": {"q": "q"}}, "observation": "evidence"}
        error = {"assistant": '<think>x</think><tool_call>{}</tool_call>', "tool_error": True}
        result = reward_parts({"trace": [good, error, error, error], "termination": "fatal_error"})
        self.assertEqual(result["valid_steps"], 1)
        self.assertEqual(result["format_reward"], 1)
        self.assertEqual(result["terminal_answer"], "")
        normal = reward_parts({"trace": [{"assistant": '<think>x</think><response>54</response>'}], "termination": "final_answer"})
        self.assertEqual(normal["terminal_answer"], "54")

    def test_group_is_incomplete_until_all_four_rewards_exist(self):
        import importlib.util

        self.assertIsNotNone(importlib.util.find_spec("rl_diagnostic"))
        from rl_diagnostic import group_summary

        self.assertFalse(group_summary([{"accuracy": 1, "reward": 0.9}]) ["complete"])
        result = group_summary([{"accuracy": 0, "reward": x} for x in (0.02, 0.04, 0.06, 0.08)])
        self.assertTrue(result["complete"])
        self.assertFalse(result["equal_rewards"])
        self.assertEqual(result["correct_count"], 0)
