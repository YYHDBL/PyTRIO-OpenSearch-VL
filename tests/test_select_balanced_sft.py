"""分层选择只统计实际工具调用，并排除不完整或重复的轨迹。"""

import importlib.util
import unittest


class BalancedSFTTest(unittest.TestCase):
    def features(self, replies):
        self.assertIsNotNone(importlib.util.find_spec("select_balanced_sft"))
        from select_balanced_sft import trajectory_features
        return trajectory_features({"images": ["one.png"], "conversations": [
            {"from": "human", "value": "<image>question"},
            *({"from": "gpt", "value": r} for r in replies)]})

    def test_only_executable_calls_define_the_tool_family(self):
        f = self.features(['<think>Example <tool_call>{"name":"sharpen","arguments":{}}</tool_call></think>'
                           '<tool_call>{"name":"image_search","arguments":{"url":"img_1"}}</tool_call>',
                           '<response>answer</response>'])
        self.assertEqual(f["family"], "retrieval")
        self.assertEqual(f["round_bin"], "short")
        self.assertEqual(f["tools"], {"image_search"})

    def test_visual_tools_are_multilabel_and_enhancement_has_priority(self):
        calls = ['<tool_call>{"name":"crop","arguments":{"image":"img_1"}}</tool_call>',
                 '<tool_call>{"name":"layout_parsing","arguments":{"image":"img_2"}}</tool_call>',
                 '<tool_call>{"name":"super_resolution","arguments":{"image":"img_2"}}</tool_call>']
        f = self.features(calls + ['<response>answer</response>'])
        self.assertEqual(f["family"], "enhancement")
        self.assertEqual(f["round_bin"], "medium")
        self.assertEqual(len(f["tools"]), 3)

    def test_unfinished_answers_and_repeated_identical_actions_are_rejected(self):
        call = '<tool_call>{"name":"text_search","arguments":{"q":"same"}}</tool_call>'
        self.assertIsNone(self.features([call, '<think>unfinished <response>candidate</response>']))
        self.assertIsNone(self.features([call] * 3 + ['<response>answer</response>']))

    def test_question_decontamination_removes_only_the_known_image_metadata_prefix(self):
        self.assertIsNotNone(importlib.util.find_spec("select_balanced_sft"))
        from select_balanced_sft import normalized_question
        self.assertEqual(normalized_question("<image>\nimage_id: img_1 Question: Which city?"),
                         normalized_question("Which city?"))
