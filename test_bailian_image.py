"""百炼图搜图的响应结构与异常边界。"""

import importlib.util
import json
import unittest


class BailianImageTest(unittest.TestCase):
    def test_tool_request_has_enough_output_budget_and_disables_reasoning(self):
        import bailian_image

        self.assertTrue(hasattr(bailian_image, "image_search_payload"))
        payload = bailian_image.image_search_payload("qwen3.8-27b", "https://example.org/image.png")
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertEqual(payload["max_output_tokens"], 4096)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["tools"], [{"type": "image_search"}])
        self.assertIn("reply only OK", payload["input"][0]["content"][0]["text"])

    def test_search_results_and_usage_are_extracted(self):
        self.assertIsNotNone(importlib.util.find_spec("bailian_image"))
        from bailian_image import parse_search_response

        response = {"status": "completed", "model": "qwen3.8-27b",
                    "output": [{"type": "image_search_call", "status": "completed", "arguments": '{"img_idx":0,"bbox":[0,0,1000,1000]}',
                                "output": json.dumps([{"index": 1, "title": "Bridge", "url": "https://example.org/bridge.jpg"}])}],
                    "usage": {"input_tokens": 200, "output_tokens": 30, "x_tools": {"image_search": {"count": 1}}}}
        result = parse_search_response(response)
        self.assertEqual(result["matches"][0]["title"], "Bridge")
        self.assertEqual(result["usage"]["image_search_calls"], 1)
        self.assertEqual(result["usage"]["input_tokens"], 200)
        self.assertEqual(result["calls"][0]["arguments"]["bbox"], [0, 0, 1000, 1000])

    def test_response_without_a_completed_tool_call_is_not_a_search_result(self):
        self.assertIsNotNone(importlib.util.find_spec("bailian_image"))
        from bailian_image import parse_search_response

        for response in ({"status": "completed", "output": [{"type": "message"}], "usage": {}},
                         {"status": "incomplete", "output": [], "usage": {}},
                         {"status": "completed", "output": [{"type": "image_search_call", "status": "failed"}], "usage": {}}):
            with self.assertRaises(ValueError):
                parse_search_response(response)

    def test_completed_cache_hit_does_not_invoke_usage_callback(self):
        # 缓存命中不产生费用，也不应触发计费回调；只有真实调用路径才回调。
        import hashlib
        import os
        from pathlib import Path
        import shutil
        import bailian_image

        root = Path("outputs/test-bailian-cache")
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        saved = {name: os.environ.get(name) for name in
                 ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_IMAGE_MODEL")}
        os.environ["DASHSCOPE_API_KEY"] = "test-key"
        os.environ["DASHSCOPE_BASE_URL"] = "https://llm-x.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        os.environ["DASHSCOPE_IMAGE_MODEL"] = "qwen3.8-27b"
        try:
            client = bailian_image.BailianImageClient(root)
            url = "https://example.org/a.jpg"
            parameters = {"image_sha256": hashlib.sha256(url.encode()).hexdigest(), "model": client.model,
                          "base_url": client.base_url, "tool_choice": "required", "max_output_tokens": 4096,
                          "reasoning": "none", "temperature": 0, "response_mode": "tool-only-v1"}
            key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
            cached = {"matches": [], "calls": [], "response_model": "qwen3.8-27b",
                      "usage": {"input_tokens": 1, "output_tokens": 1, "image_search_calls": 1,
                                "observed_completed_calls": 1}}
            (root / f"{key}.result.json").write_text(
                json.dumps({"parameters": parameters, "result": cached}) + "\n", encoding="utf-8")
            called = []
            result = client.search(url, before_submit=lambda: None, on_result=called.append)
            self.assertEqual(result, cached)
            self.assertEqual(called, [])
        finally:
            shutil.rmtree(root, ignore_errors=True)
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_connect_failure_clears_pending_but_uncertain_failure_keeps_it(self):
        # 连接未建立：请求未送达、不计费，pending 必须删除以便重试；
        # 发送后失败：结果不确定，pending 必须保留，禁止自动重传。
        import hashlib
        import os
        from pathlib import Path
        from unittest import mock
        import shutil
        import httpx as httpx_module
        import bailian_image

        root = Path("outputs/test-bailian-pending")
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        saved = {name: os.environ.get(name) for name in
                 ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_IMAGE_MODEL")}
        os.environ["DASHSCOPE_API_KEY"] = "test-key"
        os.environ["DASHSCOPE_BASE_URL"] = "https://llm-x.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        os.environ["DASHSCOPE_IMAGE_MODEL"] = "qwen3.8-27b"
        try:
            client = bailian_image.BailianImageClient(root)
            url = "https://example.org/b.jpg"
            parameters = {"image_sha256": hashlib.sha256(url.encode()).hexdigest(), "model": client.model,
                          "base_url": client.base_url, "tool_choice": "required", "max_output_tokens": 4096,
                          "reasoning": "none", "temperature": 0, "response_mode": "tool-only-v1"}
            key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()

            class ConnectFail:
                def __init__(self, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def post(self, *args, **kwargs):
                    raise httpx_module.ConnectError("no route")

            with mock.patch.object(bailian_image.httpx, "Client", ConnectFail):
                with self.assertRaises(bailian_image.BailianImageError) as ctx:
                    client.search(url, before_submit=lambda: None)
            self.assertIn("未发送", str(ctx.exception))
            self.assertFalse((root / f"{key}.pending.json").exists())

            class ReadTimeoutFail(ConnectFail):
                def post(self, *args, **kwargs):
                    raise httpx_module.ReadTimeout("uncertain")

            with mock.patch.object(bailian_image.httpx, "Client", ReadTimeoutFail):
                with self.assertRaises(bailian_image.BailianImageError):
                    client.search(url, before_submit=lambda: None)
            self.assertTrue((root / f"{key}.pending.json").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


if __name__ == "__main__":
    unittest.main()
