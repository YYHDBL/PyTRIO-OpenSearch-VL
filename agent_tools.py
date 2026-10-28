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
import cv2
import numpy as np
from PIL import Image

from doubao_image import DoubaoImageClient, DoubaoImageError
from deepseek_search import DeepSeekPageUnavailable, DeepSeekSearchClient, DeepSeekServiceError
from paddle_ocr import OCRServiceError, PaddleOCRClient


SUPPORTED_TOOLS = {"crop", "layout_parsing", "web_search", "image_search", "text_search",
                   "perspective_correct", "super_resolution", "sharpen"}


def text_summary_prompt(content: str, query: str, title: str) -> str:
    return (f'Based on the following webpage content, provide a concise summary that is relevant to the query: "{query}"\n\n'
            f"Webpage Title: {title}\nContent:\n{content[:2000]}\n\n"
            "Please provide a focused summary (2-4 sentences) that directly addresses the query. Focus on the most relevant information.")


def image_summary_prompt(result: dict) -> str:
    value = json.dumps(result, ensure_ascii=False, indent=2)
    return ('You are processing image search results. Extract and summarize only the relevant "title" and "source" information '
            'from the following image search results. Remove all irrelevant information and keep only the essential identification details.\n\n'
            f"Image Search Results:\n{value[:3000]}\n\n"
            'Please extract and return ONLY the relevant information in JSON format with "title" and "source" fields. '
            'If there are multiple results, return a list of objects, each with "title" and "source" fields. '
            'Remove any irrelevant details, descriptions, or metadata that are not directly related to identifying the object/entity.')


class BudgetExhausted(RuntimeError):
    pass


class ToolError(RuntimeError):
    pass


class PageUnavailable(ToolError):
    pass


def model_image_bytes(image: Image.Image) -> bytes:
    rgba = image.convert("RGBA")
    rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    output = io.BytesIO()
    rgb.save(output, format="PNG")
    return output.getvalue()


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
    def __init__(self, cache_dir: Path, image_limit: int = 400, text_limit: int = 1500,
                 summarizer=None, summary_model: str = "", ocr_cache_dir: Path | None = None,
                 usage_dir: Path | None = None):
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = cache_dir / "requests.jsonl"
        self.image_limit = image_limit
        self.text_limit = text_limit
        self.super_resolution = None
        self.summarizer, self.summary_model = summarizer, summary_model
        self.ocr_cache_dir = ocr_cache_dir if ocr_cache_dir is not None else cache_dir / "ocr-v1"
        self.usage_dir = usage_dir
        self.stats = {"text_search": 0, "image_search": 0, "reader": 0, "layout_parsing": 0, "cache_hit": 0}

    def record_bailian_usage(self, result: dict) -> None:
        # 仅真实计费成功的请求进入账本；缓存命中不产生费用，不记录。
        if self.usage_dir is None:
            return
        row = {"at": time.time(), "kind": "image_search", **result["usage"],
               "response_model": result["response_model"], "response_id": result.get("response_id"),
               "response_status": result.get("response_status")}
        with (self.usage_dir / "bailian-usage.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def record_doubao_usage(self, result: dict) -> None:
        if self.usage_dir is None:
            return
        row = {"at": time.time(), "kind": "image_search", "provider": result["provider"],
               "request_id": result["request_id"], "returned_doc_count": result["returned_doc_count"],
               "total_doc_count": result["total_doc_count"], "provider_error_code": result.get("provider_error_code"),
               "response_kind": result.get("response_kind")}
        with (self.usage_dir / "doubao-usage.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

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
        window, count = 60, 18
        recent = [r["at"] for r in rows if r["kind"] == kind and now - r["at"] < window]
        if kind == "reader" and len(recent) >= count:
            time.sleep(max(0, min(recent) + window + 1 - now))
            now = time.time()
        # 先记录尝试，连接中断时也不丢失可能产生的消耗。
        with self.ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now, "kind": kind}) + "\n")
        self.stats[kind] += 1

    def request(self, method: str, url: str, *, direct: bool = False, **kwargs) -> dict:
        # 仅对连接未建立的失败重试：请求必然未发出，不会重复计费或重复产生副作用；
        # 发送后的任何失败仍立即上抛，不掩盖真实服务错误。
        attempts = 3
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=45, trust_env=not direct) as client:
                    response = client.request(method, url, **kwargs)
                break
            except (httpx.ConnectError, httpx.ConnectTimeout):
                if attempt == attempts - 1:
                    raise ToolError("工具网络请求失败: 连接始终无法建立") from None
                print(f"tool-connect-retry url={url} attempt={attempt + 1}", flush=True)
                time.sleep(2 ** attempt)
            except httpx.HTTPError as exc:
                raise ToolError(f"工具网络请求失败: {type(exc).__name__}") from None
        try:
            if response.status_code != 200:
                raise ToolError(f"工具服务 HTTP {response.status_code}")
            result = response.json()
        except ValueError:
            raise ToolError("工具网络请求失败: 响应不是有效 JSON") from None
        if not isinstance(result, dict) or result.get("error"):
            raise ToolError("工具服务返回错误")
        return result

    def read_page(self, url: str) -> dict:
        validate_public_url(url)
        def fetch():
            self.charge("reader")
            try:
                result = DeepSeekSearchClient(self.cache_dir / "deepseek-http", self.usage_dir).fetch(url)
            except DeepSeekPageUnavailable as error:
                raise PageUnavailable(str(error)) from None
            except DeepSeekServiceError as error:
                raise ToolError(str(error)) from None
            return {"title": result["title"], "content": result["content"][:5000]}
        return self.cached("deepseek-page-v1", {"url": url}, fetch)

    def summarize(self, prompt: str) -> str:
        if self.summarizer is None or not self.summary_model:
            raise ToolError("七工具协议需要配置独立摘要模型")
        result = self.cached("summary-v2", {"model": self.summary_model, "prompt": prompt},
                             lambda: self.summarizer(prompt))
        if not isinstance(result, str) or not result.strip():
            raise ToolError("摘要模型未返回有效内容")
        return result.strip()

    def execute(self, action: dict, images: dict[str, bytes]) -> tuple[str, bytes | None]:
        name, args = action["name"], action["arguments"]
        if name == "layout_parsing":
            ref = args.get("image")
            if ref not in images:
                raise ValueError("版面解析需要已登记的图片引用")
            options = {"useDocOrientationClassify": args.get("use_doc_orientation_classify", False),
                       "useDocUnwarping": args.get("use_doc_unwarping", False),
                       "useChartRecognition": args.get("use_chart_recognition", False)}
            if any(type(value) is not bool for value in options.values()):
                raise ValueError("OCR 功能开关必须为布尔值")
            self.stats["layout_parsing"] += 1
            try:
                result = PaddleOCRClient(self.ocr_cache_dir).parse(images[ref], options)
            except OCRServiceError as error:
                raise ToolError(str(error)) from None
            except ValueError:
                raise ToolError("飞桨 OCR 返回数据格式无效") from None
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
            return f"Image cropped successfully. New image ID: {new_ref}.", cropped
        if name in {"sharpen", "perspective_correct", "super_resolution"}:
            ref = args.get("image")
            if ref not in images:
                raise ValueError("图片引用不存在")
            with Image.open(io.BytesIO(images[ref])) as image:
                original = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            cv2.setNumThreads(2)
            if name == "sharpen":
                amount = args.get("amount", 1.5)
                if type(amount) not in (float, int) or not np.isfinite(amount) or amount < 0:
                    raise ValueError("锐化强度必须是非负有限数值")
                blurred = cv2.GaussianBlur(original, (0, 0), 3)
                output = cv2.addWeighted(original, 1.0 + amount, blurred, -amount, 0)
                detail = "Image sharpened successfully."
            elif name == "perspective_correct":
                gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200)
                contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                corners = None
                for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                    candidate = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    if len(candidate) == 4:
                        corners = candidate.reshape(4, 2).astype(np.float32)
                        break
                if corners is None:
                    output = original
                    detail = "No quadrilateral detected; image was not transformed."
                else:
                    sums, differences = corners.sum(axis=1), np.diff(corners, axis=1).reshape(-1)
                    top_left, top_right, bottom_right, bottom_left = (corners[np.argmin(sums)], corners[np.argmin(differences)],
                                                                     corners[np.argmax(sums)], corners[np.argmax(differences)])
                    width = max(int(np.linalg.norm(bottom_right - bottom_left)), int(np.linalg.norm(top_right - top_left)))
                    height = max(int(np.linalg.norm(top_right - bottom_right)), int(np.linalg.norm(top_left - bottom_left)))
                    if width < 2 or height < 2:
                        raise ValueError("检测到的透视区域无效")
                    rect = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
                    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
                    output = cv2.warpPerspective(original, cv2.getPerspectiveTransform(rect, dst), (width, height))
                    detail = "Perspective corrected successfully."
            else:
                scale = args.get("scale", 4)
                if type(scale) is not int or scale != 4:
                    raise ValueError("当前 EDSR 权重只支持 4 倍超分辨率")
                weight = Path(os.environ.get("SR_MODEL_PATH", "data/models/EDSR_x4.pb"))
                if not weight.is_file():
                    raise FileNotFoundError(f"EDSR 权重不存在: {weight}")
                if self.super_resolution is None:
                    if weight.stat().st_size != 38573255:
                        raise ValueError("EDSR 权重文件大小不正确")
                    engine = cv2.dnn_superres.DnnSuperResImpl_create()
                    engine.readModel(str(weight))
                    engine.setModel("edsr", 4)
                    self.super_resolution = engine
                output = self.super_resolution.upsample(original)
                if output.shape[:2] != (original.shape[0] * 4, original.shape[1] * 4):
                    raise ValueError("超分辨率输出尺寸错误")
                detail = "Image enhanced by EDSR x4 successfully."
            transformed = model_image_bytes(Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)))
            new_ref = f"img_{len(images) + 1}"
            images[new_ref] = transformed
            return f"{detail} New image ID: {new_ref}.", transformed
        if name in {"text_search", "web_search"}:
            query = args.get("q") or args.get("query")
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 1000:
                raise ValueError("搜索词为空或过长")
            count = args.get("top_k", 5)
            if type(count) is not int or not 1 <= count <= 5:
                raise ValueError("top_k 必须为 1 到 5")
            # DeepSeek 没有 Google 的 hl 参数；原 query 不改写、不静默截断，语言由 query 决定。
            parameters = {"q": query.strip(), "num": count, "need_content": True}
            def search():
                self.charge("text_search")
                try:
                    return DeepSeekSearchClient(self.cache_dir / "deepseek-http", self.usage_dir).search(query, count)
                except DeepSeekServiceError as error:
                    raise ToolError(str(error)) from None
            results = self.cached("deepseek-search-v1", parameters, search)
            if not results:
                return "Tool execution result:\nNo relevant web pages found for the query.", None
            passages = []
            for index, item in enumerate(results, 1):
                content = item.get("content") or ""
                status = ""
                if not content and item.get("link"):
                    try:
                        content = self.read_page(item["link"])["content"]
                    except PageUnavailable as exc:
                        content = item.get("summary") or item.get("snippet") or ""
                        status = f"\nSource notice: webpage unavailable ({exc}); summary uses search result excerpt, not full webpage text."
                if not content:
                    raise ToolError("搜索结果没有可摘要的正文或片段")
                summary = self.summarize(text_summary_prompt(content, query, item.get("title") or ""))
                passages.append(f"[Passage {index}]\nTitle: {item.get('title', '')}\nURL: {item.get('link', '')}\nSummary:\n{summary}{status}")
            return "Tool execution result:\n" + ("\n\n" + "=" * 60 + "\n\n").join(passages), None
        if name != "image_search":
            raise ValueError("不支持的工具")
        ref = args.get("url")
        if ref in images:
            image_input = images[ref]
        elif isinstance(ref, str):
            validate_public_url(ref)
            image_input = ref
        else:
            raise ValueError("图片引用不存在")
        try:
            result = DoubaoImageClient(self.cache_dir / "doubao-image-v1").search(
                image_input, before_submit=lambda: self.charge("image_search"),
                on_result=self.record_doubao_usage)
        except DoubaoImageError as error:
            raise ToolError(str(error)) from None
        if not result["matches"]:
            return "Tool execution result:\nNo visual matches found.", None
        evidence = {"visual_matches": [{"title": item["title"],
                                        "source": item["source"] or "Source webpage unknown; image URLs: " + ", ".join(item["image_urls"])}
                                       for item in result["matches"]]}
        summary = self.summarize(image_summary_prompt(evidence))
        notice = ""
        if any(not item["source"] for item in result["matches"]):
            notice = "Source notice: some matches lack source webpage URLs; image URLs are not webpage provenance.\n"
        return "Tool execution result:\n" + notice + summary, None
