"""百炼 Responses 图搜图；缓存结果并保留无法确认的计费请求。"""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx
from PIL import Image

from service_io import save_http_response, write_json


class BailianImageError(RuntimeError):
    pass


def image_search_payload(model: str, image_url: str) -> dict:
    # 只需要检索结果；关闭额外思考，避免模型解说耗尽预算使整段响应 incomplete。
    return {"model": model, "tools": [{"type": "image_search"}], "tool_choice": "required",
            "parallel_tool_calls": False, "max_output_tokens": 4096, "store": False,
            "reasoning": {"effort": "none"}, "temperature": 0,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Use image_search exactly once to find visually similar images. After the tool result is available, reply only OK. The application reads the tool result directly."},
                {"type": "input_image", "image_url": image_url}]}]}


def response_usage(response: dict) -> dict:
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        raise ValueError("百炼用量字段无效")
    x_tools = usage.get("x_tools") or {}
    if not isinstance(x_tools, dict):
        raise ValueError("百炼工具用量字段无效")
    tool_info = x_tools.get("image_search", {})
    reported = tool_info.get("count") if isinstance(tool_info, dict) else None
    if reported is not None and (type(reported) is not int or reported < 0):
        raise ValueError("百炼图搜图调用次数无效")
    calls = response.get("output", [])
    if not isinstance(calls, list):
        raise ValueError("百炼输出不是数组")
    return {"usage": {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                      "image_search_calls": reported,
                      "observed_completed_calls": sum(isinstance(r, dict) and r.get("type") == "image_search_call"
                                                      and r.get("status") == "completed" for r in calls)},
            "response_model": response.get("model", ""), "response_id": response.get("id"),
            "response_status": response.get("status")}


def parse_search_response(response: dict) -> dict:
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise ValueError("百炼响应未完成")
    if not isinstance(response.get("output"), list):
        raise ValueError("百炼输出不是数组")
    calls = [item for item in response["output"]
             if isinstance(item, dict) and item.get("type") == "image_search_call"]
    if not calls or any(item.get("status") != "completed" for item in calls):
        raise ValueError("百炼未完成图搜图工具调用")
    matches, call_details = [], []
    for item in calls:
        arguments = item.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if arguments is not None and not isinstance(arguments, dict):
            raise ValueError("百炼图搜图参数格式无效")
        call_details.append({"arguments": arguments})
        raw = item.get("output")
        if not isinstance(raw, str):
            raise ValueError("百炼图搜图输出不是 JSON 字符串")
        decoded = json.loads(raw)
        if not isinstance(decoded, list):
            raise ValueError("百炼图搜图输出不是结果列表")
        for match in decoded:
            if not isinstance(match, dict) or not isinstance(match.get("url"), str):
                raise ValueError("百炼图搜图结果缺少图片 URL")
            matches.append({"index": match.get("index"), "title": match.get("title", ""), "url": match["url"]})
    return {"matches": matches, "calls": call_details, **response_usage(response)}


class BailianImageClient:
    def __init__(self, cache_dir: Path):
        self.key = os.environ["DASHSCOPE_API_KEY"]
        self.base_url = os.environ["DASHSCOPE_BASE_URL"].rstrip("/")
        self.model = os.environ.get("DASHSCOPE_IMAGE_MODEL", "qwen3.8-27b")
        parsed = urlsplit(self.base_url)
        allowed_host = (parsed.hostname or "").endswith(".maas.aliyuncs.com") or parsed.hostname in {
            "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"
        }
        if (not self.key.strip() or parsed.scheme != "https" or not parsed.hostname
                or not allowed_host or parsed.query or parsed.fragment
                or parsed.path.rstrip("/") != "/compatible-mode/v1" or parsed.username or parsed.password):
            raise ValueError("百炼需要有效密钥和业务空间 HTTPS Base URL")
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def search(self, image_input: bytes | str, before_submit, on_result=None) -> dict:
        if isinstance(image_input, bytes):
            if len(image_input) > 7_000_000:
                raise ValueError("图片超过百炼 Base64 输入大小上限，请先裁剪")
            with Image.open(io.BytesIO(image_input)) as image:
                mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(image.format)
                image.verify()
            if not mime:
                raise ValueError("百炼不支持该图片格式")
            image_url = "data:" + mime + ";base64," + base64.b64encode(image_input).decode("ascii")
            image_hash = hashlib.sha256(image_input).hexdigest()
        elif isinstance(image_input, str):
            parsed = urlsplit(image_input)
            host = (parsed.hostname or "").casefold()
            if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
                raise ValueError("图片 URL 无效")
            if host == "localhost" or "." not in host or host.endswith((".local", ".internal")):
                raise ValueError("图片 URL 必须可公网访问")
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                address = None
            if address is not None and not address.is_global:
                raise ValueError("图片 URL 必须可公网访问")
            image_url = image_input
            image_hash = hashlib.sha256(image_input.encode()).hexdigest()
        else:
            raise TypeError("图片输入必须为图片字节或公网 URL")
        parameters = {"image_sha256": image_hash, "model": self.model,
                      "base_url": self.base_url, "tool_choice": "required", "max_output_tokens": 4096,
                      "reasoning": "none", "temperature": 0, "response_mode": "tool-only-v1"}
        key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        completed = self.cache_dir / f"{key}.result.json"
        pending = self.cache_dir / f"{key}.pending.json"
        if completed.exists():
            return json.loads(completed.read_text(encoding="utf-8"))["result"]
        if pending.exists():
            raise BailianImageError("上次百炼请求结果未确认，请先检查账单与服务状态，避免重复计费")
        payload = image_search_payload(self.model, image_url)
        before_submit()
        write_json(pending, {"state": "submitted", "at": time.time(), "parameters": parameters})
        try:
            with httpx.Client(timeout=httpx.Timeout(240, connect=10), trust_env=False) as client:
                response = client.post(self.base_url + "/responses",
                                       headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
                                       json=payload)
            save_http_response(self.cache_dir / f"{key}.http.json", response)
            if response.status_code != 200:
                raise BailianImageError(f"百炼请求 HTTP {response.status_code}；提交记录已保留")
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("百炼响应不是对象")
            # incomplete 响应也可能已执行检索；先记服务端报告的用量，再做完整性校验。
            if on_result is not None and body.get("usage"):
                on_result(response_usage(body))
            result = parse_search_response(body)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            # 连接未建立，请求必然未送达、不会计费；删除提交标记后允许安全重试。
            pending.unlink(missing_ok=True)
            raise BailianImageError("百炼连接未建立，请求未发送，可安全重试") from None
        except (httpx.HTTPError, ValueError) as error:
            detail = str(error) if isinstance(error, ValueError) else type(error).__name__
            record = json.loads(pending.read_text())
            record.update(error_type=type(error).__name__, response_received=(self.cache_dir / f"{key}.http.json").exists())
            write_json(pending, record)
            raise BailianImageError(f"百炼图搜图失败: {detail}；提交标记保留，已收到的回包另存证据") from None
        write_json(completed, {"parameters": parameters, "result": result, "at": time.time()})
        pending.unlink()
        return result
