"""OCR 结果的真实字段解析与输入边界检查。"""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import unittest


class PaddleOcrTest(unittest.TestCase):
    def test_invalid_explicit_proxy_fails_before_submission(self):
        from paddle_ocr import OCRServiceError, PaddleOCRClient

        client = PaddleOCRClient.__new__(PaddleOCRClient)
        client.token = "test-only-token"
        previous = os.environ.get("PADDLE_OCR_PROXY")
        os.environ["PADDLE_OCR_PROXY"] = "invalid://127.0.0.1:0"
        try:
            with self.assertRaises((ValueError, OCRServiceError)) as raised:
                client.request("POST", "https://127.0.0.1:0")
            self.assertIsInstance(raised.exception, ValueError)
        finally:
            if previous is None:
                os.environ.pop("PADDLE_OCR_PROXY", None)
            else:
                os.environ["PADDLE_OCR_PROXY"] = previous

    def test_unknown_submission_outcome_blocks_automatic_reupload(self):
        import paddle_ocr

        self.assertTrue(hasattr(paddle_ocr, "assert_submission_safe"))
        root = Path("outputs/test-ocr-uncertain")
        self.assertFalse(root.exists())
        root.mkdir(parents=True)
        try:
            marker = root / "submission.json"
            paddle_ocr.assert_submission_safe(marker)
            marker.write_text('{"state":"submitting"}')
            with self.assertRaises(paddle_ocr.OCRServiceError):
                paddle_ocr.assert_submission_safe(marker)
        finally:
            shutil.rmtree(root)

    def test_layout_blocks_use_the_author_text_label_filter(self):
        self.assertIsNotNone(importlib.util.find_spec("paddle_ocr"))
        from paddle_ocr import normalize_results

        blocks = [{"block_label": "paragraph_title", "block_content": " Title "},
                  {"block_label": "text", "block_content": "Body"},
                  {"block_label": "table", "block_content": "Table"},
                  {"block_label": "vision_footnote", "block_content": "Note"}]
        line = json.dumps({"result": {"layoutParsingResults": [
            {"prunedResult": {"parsing_res_list": blocks}, "markdown": {"text": "whole page"}}]}})
        result = normalize_results(line)
        self.assertEqual(result["rec_texts"], ["Title", "Body", "Note"])
        self.assertEqual(result["formatted_text"], "Title\nBody\nNote")
        self.assertEqual(result["blocks"], blocks)

    def test_invalid_results_are_not_silently_empty_ocr(self):
        self.assertIsNotNone(importlib.util.find_spec("paddle_ocr"))
        from paddle_ocr import normalize_results

        for invalid in ("", "[]", '{"error":"service failed"}',
                        '{"result":{"layoutParsingResults":[{}]}}',
                        '{"result":{"layoutParsingResults":[[]]}}'):
            with self.assertRaises(ValueError):
                normalize_results(invalid)

    def test_private_and_non_https_download_urls_are_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec("paddle_ocr"))
        from paddle_ocr import validate_result_url

        for url in ("http://example.com/x", "https://127.0.0.1/x", "https://localhost/x", "file:///etc/passwd"):
            with self.assertRaises(ValueError):
                validate_result_url(url)
        validate_result_url("https://example.com/results.jsonl")


if __name__ == "__main__":
    unittest.main()
