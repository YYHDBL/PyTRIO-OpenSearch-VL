"""作者答案抽取口径：完整轨迹、标签优先级和未闭合思考回退。"""

import importlib.util
import unittest


class AnswerExtractionTest(unittest.TestCase):
    def test_unclosed_thinking_is_not_automatically_empty(self):
        self.assertIsNotNone(importlib.util.find_spec("answer_extraction"))
        from answer_extraction import extract_final_answer

        text = "<think>Still investigating the candidate entity."
        self.assertEqual(extract_final_answer(text), text)

    def test_upstream_tag_precedence_and_nested_box(self):
        self.assertIsNotNone(importlib.util.find_spec("answer_extraction"))
        from answer_extraction import extract_final_answer

        text = r"\boxed{\text{A{B}}} <answer>second</answer><response>third</response>"
        self.assertEqual(extract_final_answer(text), "A{B}")
        self.assertEqual(extract_final_answer("<answer>second</answer><response>third</response>"), "second")

    def test_full_trace_is_used_and_tool_text_is_filtered_after_thinking(self):
        self.assertIsNotNone(importlib.util.find_spec("answer_extraction"))
        from answer_extraction import evaluation_response

        record = {"answer": "", "trace": [{"assistant": "<answer>earlier answer</answer>"},
                                          {"assistant": "<think>later unfinished thought"}]}
        self.assertEqual(evaluation_response(record), "earlier answer")


if __name__ == "__main__":
    unittest.main()
