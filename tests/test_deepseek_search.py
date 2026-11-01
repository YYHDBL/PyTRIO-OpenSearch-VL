"""DeepSeek 请求边界与错误响应不能冒充有效搜索结果。"""

import importlib.util
import unittest


class DeepSeekSearchTest(unittest.TestCase):
    def test_query_and_result_count_are_checked_without_truncation(self):
        self.assertIsNotNone(importlib.util.find_spec("deepseek_search"))
        from deepseek_search import search_payload

        for query, count in (("", 5), (" " * 3, 5), ("中" * 129, 5), ("query", 0), ("query", True)):
            with self.assertRaises(ValueError):
                search_payload(query, count)
        query = "中" * 128
        self.assertEqual(search_payload(query, 5)["query"], query)

    def test_malformed_response_is_a_service_failure(self):
        self.assertIsNotNone(importlib.util.find_spec("deepseek_search"))
        from deepseek_search import DeepSeekServiceError, parse_fetch, parse_search

        for response in ({}, [], {"results": ["invalid"]}, {"results": [{"title": "x"}]}):
            with self.assertRaises(DeepSeekServiceError):
                parse_search(response)
        for response in ({}, [], {"content": ""}, {"content": 1}):
            with self.assertRaises(DeepSeekServiceError):
                parse_fetch(response)


if __name__ == "__main__":
    unittest.main()
