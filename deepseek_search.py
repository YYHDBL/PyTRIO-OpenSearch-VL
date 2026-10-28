"""直连 DeepSeek 的独立 Search/Fetch API，保留回包并限制安全重试。"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

from service_io import save_http_response, write_json


BASE_URL = "https://api.deepseek.com/v1/tools"


class DeepSeekServiceError(RuntimeError):
    pass


class DeepSeekPageUnavailable(DeepSeekServiceError):
    pass


def search_payload(query: str, count: int) -> dict:
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 128:
        raise ValueError("DeepSeek 搜索词必须为 1–128 个字符；不会自动截断")
    if type(count) is not int or not 1 <= count <= 50:
        raise ValueError("DeepSeek 返回条数必须为 1–50 的整数")
    return {"query": query.strip(), "count": count, "need_content": True}


def parse_search(body: dict) -> list[dict]:
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise DeepSeekServiceError("DeepSeek 搜索响应缺少 results 数组")
    results = []
    for item in body["results"]:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ("title", "url")):
            raise DeepSeekServiceError("DeepSeek 搜索结果缺少标题或 URL")
        fields = {k: item.get(k, "") for k in ("title", "url", "snippet", "summary", "content")}
        if any(not isinstance(value, str) for value in fields.values()):
            raise DeepSeekServiceError("DeepSeek 搜索结果字段类型无效")
        results.append({"title": fields["title"], "link": fields["url"], "snippet": fields["snippet"],
                        "summary": fields["summary"], "content": fields["content"]})
    return results


def parse_fetch(body: dict) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("content"), str) or not body["content"].strip():
        raise DeepSeekServiceError("DeepSeek 抓取响应没有有效正文")
    if not isinstance(body.get("title", ""), str):
        raise DeepSeekServiceError("DeepSeek 抓取响应的标题类型无效")
    return {"title": body.get("title", ""), "content": body["content"]}


class DeepSeekSearchClient:
    def __init__(self, cache_dir: Path, usage_dir: Path | None = None):
        self.key = os.environ["DEEPSEEK_API_KEY"]
        if not self.key.strip():
            raise ValueError("DeepSeek API Key 为空")
        self.cache_dir, self.usage_dir = cache_dir, usage_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def request(self, operation: str, payload: dict):
        key = hashlib.sha256(json.dumps({"operation": operation, "payload": payload}, sort_keys=True).encode()).hexdigest()
        pending = self.cache_dir / f"{key}.pending.json"
        if pending.exists():
            raise DeepSeekServiceError("先前 DeepSeek 请求结果未确认，请先检查原始回包，避免重复计费")
        for attempt in range(3):
            write_json(pending, {"at": time.time(), "operation": operation, "parameters": payload})
            try:
                with httpx.Client(timeout=httpx.Timeout(60, connect=10), trust_env=False) as client:
                    response = client.post(BASE_URL + "/" + operation,
                                           headers={"Authorization": "Bearer " + self.key}, json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                pending.unlink()
                if attempt == 2:
                    raise DeepSeekServiceError("DeepSeek 连接阶段重试耗尽，请求未发送") from None
                time.sleep(2 ** attempt)
                continue
            except httpx.HTTPError as error:
                raise DeepSeekServiceError(f"DeepSeek 请求状态未确认: {type(error).__name__}；提交标记已保留") from None
            artifact = self.cache_dir / f"{key}.{time.time_ns()}.http.json"
            save_http_response(artifact, response)
            if response.status_code != 200:
                pending.unlink()
                if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                if operation == "fetch" and response.status_code in {403, 404}:
                    raise DeepSeekPageUnavailable(f"DeepSeek 无法提取此网页，HTTP {response.status_code}")
                raise DeepSeekServiceError(f"DeepSeek {operation} 返回 HTTP {response.status_code}；原始回包已保存")
            try:
                body = response.json()
                result = parse_search(body) if operation == "search" else parse_fetch(body)
            except (ValueError, DeepSeekServiceError) as error:
                raise DeepSeekServiceError(f"DeepSeek 回包解析失败: {error}；原始回包及提交标记已保留") from None
            if self.usage_dir is not None:
                with (self.usage_dir / "deepseek-usage.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"at": time.time(), "operation": operation, "http_status": 200,
                                             "billable": bool(result), "response_artifact": str(artifact)}) + "\n")
            pending.unlink()
            return result
        raise DeepSeekServiceError("DeepSeek 重试耗尽")

    def search(self, query: str, count: int = 5) -> list[dict]:
        return self.request("search", search_payload(query, count))

    def fetch(self, url: str) -> dict:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold()
        if len(url) > 8192 or parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
            raise ValueError("抓取地址必须是有效的公网 HTTP/HTTPS URL")
        if host == "localhost" or "." not in host or host.endswith((".local", ".internal")):
            raise ValueError("不能抓取内网地址")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("不能抓取内网地址")
        return self.request("fetch", {"url": url})
