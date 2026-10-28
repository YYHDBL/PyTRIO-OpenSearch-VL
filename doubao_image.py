"""豆包 Global visual 图搜图：服务器直连、原始回包、缓存及用量回调。"""

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


ENDPOINT = "https://open.feedcoopapi.com/search_api/global_search"


class DoubaoImageError(RuntimeError):
    def __init__(self, message: str, *, code=None):
        super().__init__(message)
        self.code = code


def image_search_payload(image_input: bytes | str, count: int = 10) -> dict:
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("豆包图搜返回条数必须为 1–20 的整数")
    if isinstance(image_input, bytes):
        with Image.open(io.BytesIO(image_input)) as image:
            image.verify()
        # 接口要求纯 Base64；保留原图字节和坐标，不加 Data URL 前缀。
        image_query = {"ImageBase64": base64.b64encode(image_input).decode("ascii")}
    elif isinstance(image_input, str):
        parsed = urlsplit(image_input)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
            raise ValueError("查询图片必须为有效的公网 HTTP/HTTPS URL")
        if host == "localhost" or "." not in host or host.endswith((".local", ".internal")):
            raise ValueError("不能检索内网图片地址")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("不能检索内网图片地址")
        image_query = {"Url": image_input}
    else:
        raise ValueError("图片输入必须为字节或公网 URL")
    return {"SearchType": "visual", "Query": "", "ImageQuery": image_query,
            "DocCount": count, "MaxSnippetLength": 300}


def parse_search_response(body: dict) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("ResponseMetadata"), dict):
        raise DoubaoImageError("豆包 Global 响应缺少元信息")
    metadata = body["ResponseMetadata"]
    request_id = metadata.get("RequestId")
    if not isinstance(request_id, str) or not request_id:
        raise DoubaoImageError("豆包 Global 响应缺少 RequestId")
    if metadata.get("Error"):
        error = metadata["Error"]
        code = error.get("CodeN", error.get("Code")) if isinstance(error, dict) else None
        raise DoubaoImageError(f"豆包 Global 接口拒绝请求，code={code}，RequestId={request_id}", code=code)
    result = body.get("Result")
    if not isinstance(result, dict) or type(result.get("ErrorCode")) is not int:
        raise DoubaoImageError("豆包 Global 响应缺少 Result/ErrorCode")
    empty_search = (result["ErrorCode"] == 500 and result.get("TotalDocCount") == 0
                    and result.get("Documents") in (None, [])
                    and isinstance(result.get("ErrorMsg"), str)
                    and result["ErrorMsg"].startswith("Search engine response is empty, no results returned for this query."))
    if result["ErrorCode"] != 0 and not empty_search:
        code = result["ErrorCode"]
        raise DoubaoImageError(f"豆包 Global 检索失败，code={code}，RequestId={request_id}", code=code)
    total = result.get("TotalDocCount")
    if type(total) is not int or total < 0:
        raise DoubaoImageError("豆包 Global 结果数量字段无效")
    documents = result.get("Documents")
    if documents is None and total == 0:
        documents = []
    if not isinstance(documents, list):
        raise DoubaoImageError("豆包 Global 响应缺少结果数组")
    matches = []
    for document in documents:
        if not isinstance(document, dict):
            raise DoubaoImageError("豆包图搜结果不是对象")
        title, source = document.get("Title") or "", document.get("Url") or ""
        snippets = document.get("Snippet") or []
        if not isinstance(title, str) or not isinstance(source, str) or not isinstance(snippets, list):
            raise DoubaoImageError("豆包图搜标题、来源或摘要字段无效")
        image_urls, text = [], []
        for snippet in snippets:
            if not isinstance(snippet, dict):
                raise DoubaoImageError("豆包图搜摘要不是对象")
            if snippet.get("Type") == "image":
                image = snippet.get("Image")
                if not isinstance(image, dict) or not isinstance(image.get("ImageUrl"), str):
                    raise DoubaoImageError("豆包图搜图片字段无效")
                if image["ImageUrl"]:
                    image_urls.append(image["ImageUrl"])
            elif snippet.get("Type") == "text":
                if not isinstance(snippet.get("Text"), str):
                    raise DoubaoImageError("豆包图搜文本摘要字段无效")
                text.append(snippet["Text"])
        if not source and not image_urls:
            raise DoubaoImageError("豆包图搜结果既无来源网页也无图片地址")
        matches.append({"title": title, "source": source, "image_urls": image_urls, "snippet": "\n".join(text)})
    return {"matches": matches, "request_id": request_id, "total_doc_count": total,
            "returned_doc_count": len(matches), "provider": "doubao-global-visual",
            "provider_error_code": result["ErrorCode"], "response_kind": "empty_search" if not matches else "results"}


class DoubaoImageClient:
    def __init__(self, cache_dir: Path):
        self.key = os.environ["DOUBAO_SEARCH_API_KEY"]
        if not self.key.strip():
            raise ValueError("豆包 Global API Key 为空")
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def search(self, image_input: bytes | str, before_submit, on_result=None, count: int = 10) -> dict:
        payload = image_search_payload(image_input, count)
        image_hash = hashlib.sha256(image_input if isinstance(image_input, bytes) else image_input.encode()).hexdigest()
        parameters = {"image_sha256": image_hash, "endpoint": ENDPOINT, "SearchType": "visual",
                      "DocCount": count, "MaxSnippetLength": payload["MaxSnippetLength"]}
        key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        completed = self.cache_dir / f"{key}.result.json"
        pending = self.cache_dir / f"{key}.pending.json"
        if completed.exists():
            return json.loads(completed.read_text())["result"]
        if pending.exists():
            raise DoubaoImageError("先前豆包图搜请求状态未确认，请先核对原始回包，避免重复调用")
        before_submit()
        write_json(pending, {"at": time.time(), "state": "submitting", "parameters": parameters})
        for attempt in range(3):
            try:
                with httpx.Client(timeout=httpx.Timeout(90, connect=10), trust_env=False) as client:
                    response = client.post(ENDPOINT, headers={"Authorization": "Bearer " + self.key}, json=payload)
                break
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if attempt == 2:
                    pending.unlink()
                    raise DoubaoImageError("豆包图搜连接未建立，请求未发送") from None
                time.sleep(2 ** attempt)
            except httpx.HTTPError as error:
                write_json(pending, {"at": time.time(), "parameters": parameters,
                                     "state": "uncertain", "error_type": type(error).__name__})
                raise DoubaoImageError(f"豆包图搜请求结果未确认: {type(error).__name__}；pending 已保留") from None
        artifact = self.cache_dir / f"{key}.http.json"
        save_http_response(artifact, response)
        if response.status_code != 200:
            raise DoubaoImageError(f"豆包图搜返回 HTTP {response.status_code}；原始回包与 pending 已保留")
        try:
            result = parse_search_response(response.json())
        except DoubaoImageError as error:
            if error.code is not None:
                # 服务明确拒绝的请求可在权限或额度修复后重试，未知状态仍保留 pending。
                pending.unlink()
            raise
        except ValueError:
            raise DoubaoImageError("豆包图搜回包不是有效 JSON；原始回包与 pending 已保留") from None
        if on_result is not None:
            on_result(result)
        write_json(completed, {"at": time.time(), "parameters": parameters, "result": result,
                               "response_artifact": str(artifact)})
        pending.unlink()
        return result
