"""豆包 Global 图搜图的请求边界、业务错误和来源字段。"""

import base64
import importlib.util
import io
import unittest

from PIL import Image


class DoubaoImageTest(unittest.TestCase):
    def test_base64_payload_uses_visual_and_keeps_original_image_bytes(self):
        self.assertIsNotNone(importlib.util.find_spec("doubao_image"))
        from doubao_image import image_search_payload

        stream = io.BytesIO()
        Image.new("RGB", (8, 8)).save(stream, format="PNG")
        raw = stream.getvalue()
        payload = image_search_payload(raw)
        self.assertEqual(payload["SearchType"], "visual")
        self.assertEqual(payload["Query"], "")
        self.assertEqual(base64.b64decode(payload["ImageQuery"]["ImageBase64"]), raw)
        self.assertNotIn("model", payload)
        self.assertNotIn("Url", payload["ImageQuery"])

    def test_private_urls_and_invalid_counts_are_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec("doubao_image"))
        from doubao_image import image_search_payload

        for url in ("file:///etc/passwd", "https://127.0.0.1/image.png", "http://localhost/image.png",
                    "https://user:password@example.org/image.png"):
            with self.assertRaises(ValueError):
                image_search_payload(url)
        for count in (0, 21, True):
            with self.assertRaises(ValueError):
                image_search_payload("https://example.org/image.png", count)

    def test_api_errors_are_not_treated_as_empty_searches(self):
        self.assertIsNotNone(importlib.util.find_spec("doubao_image"))
        from doubao_image import DoubaoImageError, parse_search_response

        for body in ({}, [], {"ResponseMetadata": {"RequestId": "id", "Error": {"Code": "10403"}}, "Result": None},
                     {"ResponseMetadata": {"RequestId": "id"}, "Result": {"ErrorCode": 10412}},
                     {"ResponseMetadata": {"RequestId": "id"}, "Result": {"ErrorCode": 0, "TotalDocCount": 1}}):
            with self.assertRaises(DoubaoImageError):
                parse_search_response(body)

    def test_valid_empty_results_are_preserved(self):
        self.assertIsNotNone(importlib.util.find_spec("doubao_image"))
        from doubao_image import parse_search_response

        result = parse_search_response({"ResponseMetadata": {"RequestId": "id"},
                                        "Result": {"ErrorCode": 0, "TotalDocCount": 0, "Documents": []}})
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["request_id"], "id")

    def test_observed_empty_engine_response_preserves_the_original_code(self):
        from doubao_image import DoubaoImageError, parse_search_response

        body = {"ResponseMetadata": {"RequestId": "id"}, "Result": {"ErrorCode": 500,
                "TotalDocCount": 0, "Documents": None,
                "ErrorMsg": "Search engine response is empty, no results returned for this query. Do not call this tool again during this conversation. Use existing results or another tool."}}
        try:
            result = parse_search_response(body)
        except DoubaoImageError:
            self.fail("已确认的无匹配回包应保留空结果及原始业务码")
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["provider_error_code"], 500)
        self.assertEqual(result["response_kind"], "empty_search")

    def test_other_500_errors_are_not_converted_to_no_matches(self):
        from doubao_image import DoubaoImageError, parse_search_response

        with self.assertRaises(DoubaoImageError):
            parse_search_response({"ResponseMetadata": {"RequestId": "id"},
                                   "Result": {"ErrorCode": 500, "TotalDocCount": 0, "Documents": None,
                                              "ErrorMsg": "Internal server error"}})


if __name__ == "__main__":
    unittest.main()
