"""CPU 上执行的视觉工具、搜索缓存与调用额度记录。"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image

from paddle_ocr import OCRServiceError, PaddleOCRClient


class BudgetExhausted(RuntimeError):
    pass


class ToolError(RuntimeError):
    pass


def model_image_bytes(image: Image.Image) -> bytes:
    rgba = image.convert("RGBA")
    rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    output = io.BytesIO()
    rgb.save(output, format="JPEG", quality=95)
    return output.getvalue()


def upload_image_bytes(raw: bytes) -> bytes:
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        image.thumbnail((1200, 1200))
        for quality in (85, 70, 50, 30):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality)
            if output.tell() <= 500_000:
                return output.getvalue()
    raise ValueError("图片压缩后仍超过上传限制")


def validate_public_url(url: str) -> None:
    import ipaddress

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("网页 URL 无效")
    host = parsed.hostname.casefold()
    if host == "localhost" or "." not in host or host.endswith((".local", ".internal")):
        raise ValueError("仅允许公网网页")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("仅允许公网网页")


class ToolEnvironment:
    def __init__(self, cache_dir: Path, image_limit: int = 220, text_limit: int = 1500):
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = cache_dir / "requests.jsonl"
        self.image_limit = image_limit
        self.text_limit = text_limit
        self.stats = {"text_search": 0, "image_search": 0, "reader": 0, "layout_parsing": 0, "cache_hit": 0}

    def cached(self, kind: str, parameters: dict, producer):
        identifier = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{kind}-{identifier}.json"
        if path.exists():
            self.stats["cache_hit"] += 1
            return json.loads(path.read_text(encoding="utf-8"))["result"]
        result = producer()
        path.write_text(json.dumps({"at": time.time(), "parameters": parameters, "result": result}, ensure_ascii=False) + "\n", encoding="utf-8")
        return result

    def charge(self, kind: str) -> None:
        rows = [json.loads(line) for line in self.ledger.read_text().splitlines()] if self.ledger.exists() else []
        limit = self.image_limit if kind == "image_search" else self.text_limit
        if kind != "reader" and sum(r["kind"] == kind for r in rows) >= limit:
            raise BudgetExhausted(f"{kind} 达到本次调用上限 {limit}")
        now = time.time()
        window, count = (3600, 240) if kind == "image_search" else (60, 18)
        recent = [r["at"] for r in rows if r["kind"] == kind and now - r["at"] < window]
        if kind == "reader" and len(recent) >= count:
            time.sleep(max(0, min(recent) + window + 1 - now))
            now = time.time()
        elif kind == "image_search" and len(recent) >= count:
            raise BudgetExhausted(f"{kind} 达到保守速率上限，稍后续跑")
        # 先记录尝试，连接中断时也不丢失可能产生的消耗。
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now, "kind": kind}) + "\n")
        self.stats[kind] += 1

    def request(self, method: str, url: str, *, direct: bool = False, **kwargs) -> dict:
        try:
            with httpx.Client(timeout=45, trust_env=not direct) as client:
                response = client.request(method, url, **kwargs)
            if response.status_code != 200:
                raise ToolError(f"工具服务 HTTP {response.status_code}")
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError(f"工具网络请求失败: {type(exc).__name__}") from None
        if not isinstance(result, dict) or result.get("error"):
            raise ToolError("工具服务返回错误")
        return result

    def read_page(self, url: str) -> dict:
        validate_public_url(url)
        def fetch():
            self.charge("reader")
            headers = {"Accept": "application/json", "X-Timeout": "20", "X-Retain-Images": "none"}
            if os.environ.get("JINA_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["JINA_API_KEY"]
            result = self.request("GET", "https://r.jina.ai/" + url, headers=headers)
            data = result.get("data", {})
            content = data.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise ToolError("网页未返回正文")
            return {"title": data.get("title", ""), "content": content[:5000]}
        return self.cached("page-v1", {"url": url}, fetch)

    def execute(self, action: dict, images: dict[str, bytes]) -> tuple[str, bytes | None]:
        name, args = action["name"], action["arguments"]
        if name == "layout_parsing":
            ref = args.get("image")
            if ref not in images:
                raise ValueError("版面解析需要已登记的图片引用")
            options = {"useDocOrientationClassify": args.get("use_doc_orientation_classify", False),
                       "useDocUnwarping": args.get("use_doc_unwarping", False),
                       "useChartRecognition": args.get("use_chart_recognition", False)}
            self.stats["layout_parsing"] += 1
            try:
                result = PaddleOCRClient(self.cache_dir / "ocr-v1").parse(images[ref], options)
            except OCRServiceError as error:
                raise ToolError(str(error)) from None
            body = result["formatted_text"] or "(no text detected)"
            return "Tool execution result:\nLayout Parsing SUCCESS.\n\nALL RECOGNIZED TEXT:\n" + body, None
        if name == "crop":
            ref = args.get("image")
            if ref not in images:
                raise ValueError("图片引用不存在")
            coordinates = [args[k] for k in ("x", "y", "width", "height")]
            if any(type(v) is not int for v in coordinates):
                raise ValueError("裁剪坐标必须是整数")
            x, y, width, height = coordinates
            with Image.open(io.BytesIO(images[ref])) as image:
                if min(x, y) < 0 or min(width, height) < 1 or x + width > image.width or y + height > image.height:
                    raise ValueError("裁剪区域超出图片边界")
                cropped = model_image_bytes(image.crop((x, y, x + width, y + height)))
            new_ref = f"img_{len(images) + 1}"
            images[new_ref] = cropped
            return json.dumps({"image": new_ref, "width": width, "height": height}), cropped
        if name == "text_search":
            query = args.get("q") or args.get("query")
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 1000:
                raise ValueError("搜索词为空或过长")
            count = args.get("top_k", 5)
            if type(count) is not int or not 1 <= count <= 5:
                raise ValueError("top_k 必须为 1 到 5")
            parameters = {"q": query.strip(), "hl": args.get("hl", args.get("lang", "en")), "num": count}
            def search():
                self.charge("text_search")
                result = self.request("POST", "https://google.serper.dev/search", direct=True,
                                      headers={"X-API-KEY": os.environ["SERPER_API_KEY"]}, json=parameters)
                return [{k: item.get(k, "") for k in ("title", "link", "snippet")}
                        for item in result.get("organic", [])[:count]]
            results = self.cached("text-v1", parameters, search)
            enriched = []
            for index, item in enumerate(results):
                item = item.copy()
                if index < 2 and item.get("link"):
                    try:
                        item["page"] = self.read_page(item["link"])
                    except (ToolError, ValueError) as exc:
                        item["page_error"] = str(exc)
                enriched.append(item)
            return json.dumps(enriched, ensure_ascii=False), None
        if name != "image_search":
            raise ValueError("不支持的工具")
        ref = args.get("url")
        if ref not in images:
            raise ValueError("图片引用不存在")
        content = upload_image_bytes(images[ref])
        def lens():
            self.charge("image_search")
            api_key = os.environ["SERPAPI_API_KEY"]
            uploaded = self.request("POST", "https://serpapi.com/image", data={"api_key": api_key},
                                    files={"image": ("image.jpg", content, "image/jpeg")})
            if not uploaded.get("image_id"):
                raise ToolError("图片上传未返回 image_id")
            result = self.request("GET", "https://serpapi.com/search",
                                  params={"api_key": api_key, "engine": "google_lens", "image_id": uploaded["image_id"]})
            # 不缓存含请求参数及密钥的原始搜索响应。
            return {"visual_matches": [{k: item.get(k, "") for k in ("title", "link", "source")}
                                       for item in result.get("visual_matches", [])[:8]],
                    "knowledge_graph": [{k: item.get(k, "") for k in ("title", "subtitle", "link")}
                                        for item in result.get("knowledge_graph", [])[:3]]}
        result = self.cached("lens-v1", {"sha256": hashlib.sha256(content).hexdigest()}, lens)
        return json.dumps(result, ensure_ascii=False), None
