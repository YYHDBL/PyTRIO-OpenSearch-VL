"""外部服务的私有原始回包与原子 JSON 记录。"""

import hashlib
import json
import os
from pathlib import Path
import time

import httpx


def write_private(path: Path, data: bytes) -> None:
    pending = path.with_name(path.name + ".writing")
    with os.fdopen(os.open(pending, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), "wb") as stream:
        stream.write(data)
    pending.replace(path)


def write_json(path: Path, value: dict) -> None:
    write_private(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def save_http_response(path: Path, response: httpx.Response) -> None:
    try:
        elapsed = response.elapsed.total_seconds()
    except RuntimeError:
        # 没有计时信息的离线响应保留空值，不能当作零耗时。
        elapsed = None
    body = path.with_suffix(".body")
    write_private(body, response.content)
    write_json(path, {"at": time.time(), "status_code": response.status_code, "elapsed_seconds": elapsed,
                     "headers": {k: v for k, v in response.headers.items()
                                 if k.lower() in {"x-request-id", "x-dashscope-request-id", "request-id", "content-type"}},
                     "body_path": str(body), "body_sha256": hashlib.sha256(response.content).hexdigest()})
