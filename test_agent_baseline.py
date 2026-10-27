import importlib.util
import io
import unittest

from PIL import Image


class BaselineTest(unittest.TestCase):
    def test_tool_calls_and_final_answers_are_distinguished(self):
        self.assertIsNotNone(importlib.util.find_spec("agent_baseline"))
        from agent_baseline import parse_action

        self.assertEqual(
            parse_action('<think>Check</think><tool_call>{"name":"text_search","arguments":{"q":"bridge"}}</tool_call>'),
            {"name": "text_search", "arguments": {"q": "bridge"}},
        )
        self.assertIsNone(parse_action('<response>Bridge</response>'))
        with self.assertRaises(ValueError):
            parse_action('<tool_call>{"name": "missing"}</tool_call>')

    def test_tool_call_text_inside_reasoning_is_not_executed(self):
        from agent_baseline import parse_action

        self.assertIsNone(parse_action('<think>Example <tool_call>{"name":"text_search","arguments":{"q":"x"}}</tool_call></think><response>Done</response>'))

    def test_final_answer_is_extracted_without_reasoning(self):
        import agent_baseline

        self.assertTrue(hasattr(agent_baseline, "final_answer"))
        self.assertEqual(agent_baseline.final_answer('<think>secret</think><response>BRUVIK</response>'), "BRUVIK")
        self.assertEqual(agent_baseline.final_answer('<think>unfinished'), "")

    def test_original_image_dimensions_are_preserved_for_crop(self):
        import agent_baseline

        self.assertTrue(hasattr(agent_baseline, "model_image_bytes"))
        image = Image.new("RGB", (1600, 100))
        raw = agent_baseline.model_image_bytes(image)
        with Image.open(io.BytesIO(raw)) as opened:
            self.assertEqual(opened.size, (1600, 100))

    def test_repeated_identical_searches_use_disk_cache(self):
        from pathlib import Path
        import shutil
        import agent_tools

        root = Path("outputs/test-tool-cache")
        self.assertFalse(root.exists())
        root.mkdir(parents=True)
        try:
            count = []
            def produce():
                count.append(1)
                return {"title": "result"}
            cache = agent_tools.ToolEnvironment(root)
            self.assertEqual(cache.cached("text", {"q": "x"}, produce), {"title": "result"})
            self.assertEqual(cache.cached("text", {"q": "x"}, produce), {"title": "result"})
            self.assertEqual(len(count), 1)
        finally:
            shutil.rmtree(root)

    def test_judge_format_errors_do_not_count_as_incorrect(self):
        self.assertIsNotNone(importlib.util.find_spec("judge_baseline"))
        from judge_baseline import parse_judge_response

        self.assertTrue(parse_judge_response("correct: yes\nreasoning: same answer"))
        self.assertFalse(parse_judge_response("correct: no\nreasoning: wrong entity"))
        with self.assertRaises(ValueError):
            parse_judge_response("The request failed")

    def test_crop_preserves_actual_pixel_bounds(self):
        from pathlib import Path
        import shutil
        from agent_tools import ToolEnvironment, model_image_bytes

        root = Path("outputs/test-crop")
        self.assertFalse(root.exists())
        try:
            tools = ToolEnvironment(root)
            images = {"img_1": model_image_bytes(Image.new("RGB", (1600, 100)))}
            observation, cropped = tools.execute({"name": "crop", "arguments": {
                "image": "img_1", "x": 1400, "y": 10, "width": 200, "height": 90}}, images)
            with Image.open(io.BytesIO(cropped)) as opened:
                self.assertEqual(opened.size, (200, 90))
            with self.assertRaises(ValueError):
                tools.execute({"name": "crop", "arguments": {
                    "image": "img_1", "x": 1500, "y": 10, "width": 200, "height": 90}}, images)
        finally:
            shutil.rmtree(root)

    def test_halted_protocol_run_cannot_resume(self):
        from pathlib import Path
        import shutil
        import agent_baseline

        self.assertTrue(hasattr(agent_baseline, "assert_run_allowed"))
        root = Path("outputs/test-halted-run")
        self.assertFalse(root.exists())
        root.mkdir(parents=True)
        try:
            (root / "STATUS.json").write_text('{"status":"halted_pending_protocol_review"}')
            with self.assertRaises(ValueError):
                agent_baseline.assert_run_allowed(root)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
