"""验收器必须能识别错位及错误的监督范围。"""

import importlib.util
import json
from pathlib import Path
import re
import unittest

import numpy as np


class SftAuditTest(unittest.TestCase):
    def test_tool_filter_accepts_seven_tool_actions_and_rejects_unknown_actions(self):
        import select_experiment_data
        from agent_tools import SUPPORTED_TOOLS

        self.assertTrue(hasattr(select_experiment_data, "supported_calls"))
        for name in SUPPORTED_TOOLS:
            self.assertTrue(select_experiment_data.supported_calls([json.dumps({"name": name, "arguments": {}})], SUPPORTED_TOOLS))
        self.assertFalse(select_experiment_data.supported_calls([json.dumps({"name": "unknown", "arguments": {}})], SUPPORTED_TOOLS))
        self.assertFalse(select_experiment_data.supported_calls([], SUPPORTED_TOOLS))

    def test_separate_tool_schema_is_included_without_rewriting_system(self):
        from protocol import build_messages

        tool = {"type": "function", "function": {"name": "crop", "parameters": {"type": "object"}}}
        row = {"system": "Original author instructions", "tools": json.dumps([tool]),
               "images": [], "conversations": [{"from": "human", "value": "Question"},
                                                   {"from": "gpt", "value": "Answer"}]}
        messages = build_messages(row, Path.cwd())
        system = messages[0]["content"]
        self.assertTrue(system.startswith(row["system"]))
        definitions = [value for value in re.findall(r"<tools>(.*?)</tools>", system, re.S) if value.strip()]
        self.assertEqual(len(definitions), 1)
        self.assertEqual(json.loads(definitions[0]), tool)

    def test_shift_and_mask_corruption_are_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec("audit_sft"))
        from audit_sft import validate_arrays

        tokens = np.array([10, 11, 12, 13])
        supervised = np.array([0, 1, 0, 1])
        validate_arrays(tokens, supervised, np.array([11, 0, 13]), np.array([1, 0, 1]), 3)
        with self.assertRaises(ValueError):
            validate_arrays(tokens, supervised, np.array([10, 0, 12]), np.array([1, 0, 1]), 3)
        with self.assertRaises(ValueError):
            validate_arrays(tokens, supervised, np.array([11, 12, 13]), np.ones(3), 3)
        with self.assertRaises(ValueError):
            validate_arrays(tokens, supervised, np.array([0, 0, 0]), np.zeros(3), 3)


if __name__ == "__main__":
    unittest.main()
