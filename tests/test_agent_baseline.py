import importlib.util
import ast
import inspect
import io
import os
import unittest

from PIL import Image


class BaselineTest(unittest.TestCase):
    def test_service_failure_defers_one_question_without_feedback_or_stopping_batch(self):
        import base64
        import copy
        import hashlib
        import json
        from pathlib import Path
        import shutil
        import agent_baseline
        from agent_tools import ToolEnvironment, model_image_bytes
        from doubao_image import ENDPOINT

        self.assertTrue(hasattr(agent_baseline, "run_questions"))
        root = Path("outputs/test-question-isolation")
        self.assertFalse(root.exists())
        root.mkdir(parents=True)
        previous_key = os.environ.get("DOUBAO_SEARCH_API_KEY")
        os.environ["DOUBAO_SEARCH_API_KEY"] = "unused-offline-pending-check"
        try:
            raw = model_image_bytes(Image.new("RGB", (8, 8)))
            state = {"config_sha256": "config", "phase": "tool", "termination": "", "answer": "",
                     "turns": [["assistant", '<tool_call>{"name":"image_search","arguments":{"url":"img_1"}}</tool_call>', None]],
                     "images": {"img_1": base64.b64encode(raw).decode()},
                     "trace": [{"action": {"name": "image_search", "arguments": {"url": "img_1"}}}]}
            original = copy.deepcopy(state)
            agent_baseline.write_json(root / "blocked.partial.json", state)
            parameters = {"image_sha256": hashlib.sha256(raw).hexdigest(), "endpoint": ENDPOINT,
                          "SearchType": "visual", "DocCount": 10, "MaxSnippetLength": 300}
            key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
            tools = ToolEnvironment(root / "cache")
            pending = tools.cache_dir / "doubao-image-v1" / f"{key}.pending.json"
            pending.parent.mkdir(parents=True)
            pending.write_text('{"state":"uncertain"}')
            pending_bytes = pending.read_bytes()
            agent_baseline.write_json(root / "ready.json", {"sample_id": "ready", "benchmark": "test",
                "config_sha256": "config", "termination": "final_answer", "trace": [], "elapsed_seconds": 1})

            def run_one(row):
                return agent_baseline.run_question(row, None, None, None, tools, None, root,
                                                   {"sha256": "config"}, "")

            status = agent_baseline.run_questions([{"sample_id": "blocked"}, {"sample_id": "ready"}], run_one, root)
            self.assertFalse(status["complete"])
            self.assertEqual(status["completed_ids"], ["ready"])
            self.assertEqual([r["sample_id"] for r in status["deferred"]], ["blocked"])
            retained = json.loads((root / "blocked.partial.json").read_text())
            self.assertEqual(retained["turns"], original["turns"])
            self.assertEqual(retained["images"], original["images"])
            self.assertNotIn("observation", retained["trace"][-1])
            self.assertEqual(pending.read_bytes(), pending_bytes)
            self.assertFalse(tools.ledger.exists())
            complete = agent_baseline.run_questions([{"sample_id": "ready"}], run_one, root)
            self.assertTrue(complete["complete"])
            self.assertEqual(complete["completed_ids"], ["ready"])
            tools.image_limit = 0
            with self.assertRaises(agent_baseline.BudgetExhausted):
                agent_baseline.run_questions([{"sample_id": "budget"}],
                                            lambda row: tools.charge("image_search"), root)
        finally:
            if previous_key is None:
                os.environ.pop("DOUBAO_SEARCH_API_KEY", None)
            else:
                os.environ["DOUBAO_SEARCH_API_KEY"] = previous_key
            shutil.rmtree(root)

    def test_sampler_arguments_bind_to_installed_sdk_for_base_and_checkpoint(self):
        from pathlib import Path
        from types import SimpleNamespace
        import pytrio as trio
        import agent_baseline

        tree = ast.parse(Path(agent_baseline.__file__).read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "kwargs" for t in node.targets))
        expression = compile(ast.Expression(assignment.value), "sampler-arguments", "eval")
        for path in (None, "trio://training-run/sampler_weights/final"):
            arguments = eval(expression, {"args": SimpleNamespace(model="Qwen/Qwen3.5-4B", model_path=path)})
            try:
                inspect.signature(trio.ServiceClient.create_sampling_client).bind(None, **arguments)
            except TypeError as error:
                self.fail(f"评测入口未满足实际 SDK 的 sampler 参数契约：{error}")

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

    def test_all_author_tool_names_are_accepted(self):
        from agent_baseline import parse_action

        for name in ("crop", "image_search", "text_search", "web_search", "layout_parsing",
                     "sharpen", "perspective_correct", "super_resolution"):
            action = parse_action('<tool_call>{"name":"' + name + '","arguments":{}}</tool_call>')
            self.assertEqual(action["name"], name)

    def test_author_system_includes_all_declared_tools_without_rewrite(self):
        import json
        import agent_baseline

        self.assertTrue(hasattr(agent_baseline, "author_system"))
        names = ("crop", "layout_parsing", "web_search", "image_search", "text_search",
                 "perspective_correct", "super_resolution", "sharpen")
        tools = [{"type": "function", "function": {"name": name}} for name in names]
        row = {"system": "Author prompt.", "tools": json.dumps(tools)}
        result = agent_baseline.author_system([row, row])
        self.assertTrue(result.startswith(row["system"]))
        self.assertTrue(all('"name": "' + name + '"' in result for name in names))
        with self.assertRaises(ValueError):
            agent_baseline.author_system([row, {**row, "system": "Different"}])

    def test_model_images_keep_png_and_original_dimensions(self):
        from agent_tools import model_image_bytes

        raw = model_image_bytes(Image.new("RGB", (1600, 100)))
        with Image.open(io.BytesIO(raw)) as opened:
            self.assertEqual(opened.format, "PNG")
            self.assertEqual(opened.size, (1600, 100))

    def test_sharpen_and_perspective_are_real_image_operations(self):
        from pathlib import Path
        import shutil
        import numpy as np
        from agent_tools import ToolEnvironment, model_image_bytes

        root = Path("outputs/test-image-enhancement")
        self.assertFalse(root.exists())
        try:
            tools = ToolEnvironment(root)
            pixels = np.zeros((128, 128, 3), dtype=np.uint8)
            pixels[20:110, 20:110] = 130
            images = {"img_1": model_image_bytes(Image.fromarray(pixels))}
            _, sharpened = tools.execute({"name": "sharpen", "arguments": {"image": "img_1", "amount": 1.5}}, images)
            with Image.open(io.BytesIO(sharpened)) as image:
                self.assertEqual(image.size, (128, 128))
                self.assertNotEqual(image.tobytes(), pixels.tobytes())
            _, corrected = tools.execute({"name": "perspective_correct", "arguments": {"image": "img_1"}}, images)
            with Image.open(io.BytesIO(corrected)) as image:
                self.assertGreater(image.width * image.height, 0)
            old_path = os.environ.get("SR_MODEL_PATH")
            os.environ["SR_MODEL_PATH"] = str(root / "missing.pb")
            try:
                with self.assertRaises(FileNotFoundError):
                    tools.execute({"name": "super_resolution", "arguments": {"image": "img_1"}}, images)
            finally:
                if old_path is None:
                    os.environ.pop("SR_MODEL_PATH", None)
                else:
                    os.environ["SR_MODEL_PATH"] = old_path
        finally:
            shutil.rmtree(root)

    def test_text_summary_prompt_matches_author_fields_and_content_limit(self):
        import agent_tools

        self.assertTrue(hasattr(agent_tools, "text_summary_prompt"))
        prompt = agent_tools.text_summary_prompt("A" * 2001, "bridge", "Title")
        self.assertIn('query: "bridge"', prompt)
        self.assertIn("Webpage Title: Title", prompt)
        self.assertIn("A" * 2000, prompt)
        self.assertNotIn("A" * 2001, prompt)

    def test_bailian_usage_ledger_records_live_billing_fields(self):
        import json
        from pathlib import Path
        import shutil
        import agent_tools

        root = Path("outputs/test-bailian-usage")
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        try:
            tools = agent_tools.ToolEnvironment(root / "cache", usage_dir=root)
            result = {"matches": [], "calls": [], "response_model": "qwen3.8-27b",
                      "usage": {"input_tokens": 9388, "output_tokens": 2147,
                                "image_search_calls": 1, "observed_completed_calls": 1}}
            tools.record_bailian_usage(result)
            tools.record_bailian_usage(result)
            rows = [json.loads(line) for line in (root / "bailian-usage.jsonl").open()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["input_tokens"], 9388)
            self.assertEqual(rows[0]["output_tokens"], 2147)
            self.assertEqual(rows[0]["image_search_calls"], 1)
            self.assertEqual(rows[0]["kind"], "image_search")
            self.assertEqual(rows[0]["response_model"], "qwen3.8-27b")
        finally:
            shutil.rmtree(root)
        # 未配置 usage_dir 时不落盘，也不应抛错。
        spare = Path("outputs/test-bailian-usage-2")
        shutil.rmtree(spare, ignore_errors=True)
        agent_tools.ToolEnvironment(spare).record_bailian_usage(result)
        self.assertFalse((spare / "bailian-usage.jsonl").exists())
        shutil.rmtree(spare)

    def test_connect_phase_failures_retry_but_send_phase_failures_fail_fast(self):
        # 连接未建立可安全重试；请求已发出的失败立即上抛，避免重复副作用。
        from unittest import mock
        import httpx as httpx_module
        from pathlib import Path
        import shutil
        import agent_tools

        root = Path("outputs/test-connect-retry")
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        try:
            tools = agent_tools.ToolEnvironment(root)
            calls = []

            class ConnectThenSuccess:
                def __init__(self, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def request(self, method, url, **kwargs):
                    calls.append(url)
                    if len(calls) < 3:
                        raise httpx_module.ConnectError("no route to host")
                    return mock.Mock(status_code=200, json=lambda: {"ok": True})

            with mock.patch.object(agent_tools.httpx, "Client", ConnectThenSuccess):
                self.assertEqual(tools.request("POST", "https://example.org/api", direct=True), {"ok": True})
            self.assertEqual(len(calls), 3)

            class ReadTimeoutClient(ConnectThenSuccess):
                def request(self, method, url, **kwargs):
                    calls.append(url)
                    raise httpx_module.ReadTimeout("response slow")

            calls.clear()
            with mock.patch.object(agent_tools.httpx, "Client", ReadTimeoutClient):
                with self.assertRaises(agent_tools.ToolError):
                    tools.request("POST", "https://example.org/api", direct=True)
            self.assertEqual(len(calls), 1)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
