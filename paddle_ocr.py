"""飞桨异步 OCR：保留任务 ID、限制轮询，并输出作者需要的版面字段。"""

from __future__ import annotations

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


class OCRServiceError(RuntimeError):
    pass


def assert_submission_safe(marker: Path) -> None:
    if marker.exists():
        raise OCRServiceError("先前 OCR 提交未确认收到任务 ID；请先在服务平台核对，禁止自动重复上传")


def validate_result_url(url: str) -> None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        raise ValueError("OCR 结果下载地址必须是公网 HTTPS")
    if host == "localhost" or "." not in host or host.endswith((".local", ".internal")):
        raise ValueError("OCR 结果下载地址必须是公网 HTTPS")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("OCR 结果下载地址必须是公网 HTTPS")


def normalize_results(text: str) -> dict:
    blocks, pages = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("OCR JSONL 每行必须为对象")
        result = row.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("layoutParsingResults"), list):
            raise ValueError("OCR JSONL 缺少 layoutParsingResults")
        for page in result["layoutParsingResults"]:
            if not isinstance(page, dict):
                raise ValueError("OCR 页面必须为对象")
            pruned = page.get("prunedResult")
            if not isinstance(pruned, dict) or not isinstance(pruned.get("parsing_res_list"), list):
                raise ValueError("OCR 页面缺少 parsing_res_list")
            for block in pruned["parsing_res_list"]:
                if not isinstance(block, dict) or not isinstance(block.get("block_label"), str) or not isinstance(block.get("block_content"), str):
                    raise ValueError("OCR 版面块字段无效")
                blocks.append(block)
            pages += 1
    if not pages:
        raise ValueError("OCR 结果没有页面")
    # 与作者 search.layout_parsing 使用相同的文字块标签过滤规则。
    texts = [block["block_content"].strip() for block in blocks
             if block["block_label"] in {"paragraph_title", "text", "vision_footnote"}
             and block["block_content"].strip()]
    return {"rec_texts": texts, "formatted_text": "\n".join(texts), "blocks": blocks,
            "page_count": pages, "error": None}


def save_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(".writing")
    pending.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


class PaddleOCRClient:
    def __init__(self, cache_dir: Path, timeout: float = 180):
        self.url = os.environ["PADDLE_OCR_JOB_URL"].rstrip("/")
        parsed = urlsplit(self.url)
        if parsed.scheme != "https" or parsed.hostname != "paddleocr.aistudio-app.com" or parsed.path != "/api/v2/ocr/jobs":
            raise ValueError("OCR 任务地址与已确认的飞桨服务不一致")
        self.token = os.environ["PADDLE_OCR_TOKEN"]
        self.model = os.environ["PADDLE_OCR_MODEL"]
        if not self.token.strip() or timeout <= 0:
            raise ValueError("OCR Token 为空或超时设置无效")
        self.cache_dir, self.timeout = cache_dir, timeout
        cache_dir.mkdir(parents=True, exist_ok=True)

    def request(self, method: str, url: str, **kwargs) -> dict:
        attempts = 3 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=45, trust_env=False) as client:
                    response = client.request(method, url, headers={"Authorization": "bearer " + self.token}, **kwargs)
            except httpx.HTTPError as error:
                if attempt + 1 < attempts:
                    time.sleep(2 ** attempt)
                    continue
                raise OCRServiceError(f"飞桨 OCR {method} 请求失败: {type(error).__name__}") from None
            if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < attempts:
                time.sleep(2 ** attempt)
                continue
            if response.status_code != 200:
                raise OCRServiceError(f"飞桨 OCR 返回 HTTP {response.status_code}")
            try:
                body = response.json()
            except ValueError:
                raise OCRServiceError("飞桨 OCR 未返回 JSON") from None
            break
        if not isinstance(body, dict) or not isinstance(body.get("data"), dict):
            raise OCRServiceError("飞桨 OCR 返回数据格式无效")
        return body["data"]

    def parse(self, raw: bytes, options: dict) -> dict:
        if any(type(value) is not bool for value in options.values()):
            raise ValueError("OCR 功能开关必须为布尔值")
        if set(options) - {"useDocOrientationClassify", "useDocUnwarping", "useChartRecognition"}:
            raise ValueError("OCR 选项不受支持")
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        parameters = {"model": self.model, "image_sha256": hashlib.sha256(raw).hexdigest(), "options": options}
        key = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        completed = self.cache_dir / f"{key}.result.json"
        task = self.cache_dir / f"{key}.job.json"
        submitting = self.cache_dir / f"{key}.submitting.json"
        if completed.exists():
            return json.loads(completed.read_text(encoding="utf-8"))["result"]
        started = time.monotonic()
        if task.exists():
            state = json.loads(task.read_text(encoding="utf-8"))
        else:
            assert_submission_safe(submitting)
            ledger = self.cache_dir / "submissions.jsonl"
            attempts = ledger.read_text().count("\n") if ledger.exists() else 0
            if attempts >= 1000:
                raise OCRServiceError("达到本次实验 1000 次 OCR 提交上限")
            with ledger.open("a") as stream:
                stream.write(json.dumps({"at": time.time(), "image_sha256": parameters["image_sha256"], "model": self.model}) + "\n")
            with Image.open(io.BytesIO(raw)) as image:
                suffix, mime = {"JPEG": ("jpg", "image/jpeg"), "PNG": ("png", "image/png"),
                                "WEBP": ("webp", "image/webp")}[image.format]
            # 若提交已被服务端接收但响应中断，保留标记，阻止自动重复收费。
            save_json(submitting, {"state": "submitting", "parameters": parameters, "at": time.time()})
            data = self.request("POST", self.url, data={"model": self.model, "optionalPayload": json.dumps(options)},
                                files={"file": ("image." + suffix, raw, mime)})
            job_id = data.get("jobId")
            if not isinstance(job_id, str) or not job_id or any(c not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_" for c in job_id):
                raise OCRServiceError("飞桨未返回有效任务 ID")
            state = {"job_id": job_id, "parameters": parameters, "submitted_at": time.time()}
            save_json(task, state)
            submitting.unlink()
        while True:
            if time.monotonic() - started >= self.timeout:
                raise OCRServiceError("飞桨 OCR 等待超时，任务 ID 已保留，重试将继续查询")
            data = self.request("GET", self.url + "/" + state["job_id"])
            status = data.get("state")
            state.update(last_state=status, last_polled_at=time.time())
            save_json(task, state)
            if status == "failed":
                raise OCRServiceError("飞桨 OCR 任务失败，任务记录已保留")
            if status == "done":
                url = data.get("resultUrl", {}).get("jsonUrl")
                if not isinstance(url, str):
                    raise OCRServiceError("OCR 完成但缺少 JSONL 地址")
                validate_result_url(url)
                break
            if status not in {"pending", "running"}:
                raise OCRServiceError("飞桨 OCR 返回未知任务状态")
            time.sleep(min(5, max(0, self.timeout - (time.monotonic() - started))))
        # 下载链接可能属于对象存储，绝不向该域发送 OCR Token。
        try:
            with httpx.Client(timeout=45, trust_env=False) as client:
                with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        raise OCRServiceError(f"OCR 结果下载 HTTP {response.status_code}")
                    content = bytearray()
                    for chunk in response.iter_bytes():
                        content.extend(chunk)
                        if len(content) > 20 * 1024 * 1024:
                            raise OCRServiceError("OCR 文本结果超过 20 MiB")
        except httpx.HTTPError as error:
            raise OCRServiceError(f"OCR 结果下载失败: {type(error).__name__}") from None
        result = normalize_results(content.decode("utf-8"))
        result["model"] = self.model
        save_json(completed, {"parameters": parameters, "result": result, "at": time.time()})
        return result
