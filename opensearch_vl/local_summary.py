"""临时局域网摘要后端：独立缓存身份、原始回包和保守token预算。"""

import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

from .agent_tools import BudgetExhausted
from .service_io import save_http_response, write_json, write_private


class LocalSummaryError(RuntimeError):
    pass


def summary_text(body, model):
    choice = body["choices"][0]
    text = choice["message"].get("content")
    # 原PyTRIO摘要路径接受非空的length结果；保留相同1024-token上限和文本。
    if (body.get("model") != model or choice.get("finish_reason") not in {"stop", "length"}
            or not isinstance(text, str) or not text.strip()):
        raise LocalSummaryError("局域网摘要模型、终止状态或正文无效，保留原始响应")
    return text.strip()


def configuration():
    base = os.environ["LOCAL_SUMMARY_BASE_URL"].rstrip("/")
    target = (os.environ.get("LOCAL_SUMMARY_REQUEST_BASE_URL") or base).rstrip("/")
    for url in [base, target]:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError("摘要接口必须是无内嵌凭据的HTTP地址")
    return {"backend": "local-openai", "model": os.environ["LOCAL_SUMMARY_MODEL"],
            "base_url": base, "request_base_url": target,
            "parameters": {"max_tokens": 1024, "temperature": 0.3, "top_p": 0.95, "top_k": 20,
                           "seed": 42, "chat_template_kwargs": {"enable_thinking": False}}}


def cache_identity(config):
    return "local-openai:" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


class LocalSummaryClient:
    def __init__(self, output: Path, config: dict, key: str, token_budget: int):
        if not key.strip() or token_budget < 1:
            raise ValueError("局域网摘要需要非空Key和正数预算")
        self.output, self.config, self.key, self.token_budget = output, config, key, token_budget
        self.artifacts = output / "local-summary"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.ledger = output / "local-summary-tokens.jsonl"

    def request_key(self, prompt):
        return hashlib.sha256(json.dumps({"config": self.config, "prompt": prompt}, sort_keys=True).encode()).hexdigest()

    def pending_path(self, prompt):
        return self.artifacts / (self.request_key(prompt) + ".pending.json")

    def complete(self, prompt, sample_id):
        key = self.request_key(prompt)
        completed = self.artifacts / (key + ".result.json")
        pending = self.pending_path(prompt)
        if completed.exists():
            return json.loads(completed.read_text())["text"]
        if pending.exists():
            raise LocalSummaryError("局域网摘要存在未确认请求，保留断点，不重复提交")
        rows = [json.loads(line) for line in self.ledger.read_text().splitlines()] if self.ledger.exists() else []
        # 请求前按32k完整窗口保守预留，取得服务实际usage后再据实记账。
        if sum(r.get("tokens", r["reserved_tokens"]) for r in rows) + 32768 > self.token_budget:
            raise BudgetExhausted("达到本轮局域网摘要token保护上限")
        entry = {"request_key": key, "sample_id": sample_id, "at": time.time(), "reserved_tokens": 32768,
                 "reservation_kind": "full_context_upper_bound", "backend": self.config["backend"]}
        rows.append(entry)
        write_private(self.ledger, "".join(json.dumps(row) + "\n" for row in rows).encode())
        write_json(pending, {"request_key": key, "sample_id": sample_id, "at": entry["at"]})
        payload = {"model": self.config["model"], "messages": [{"role": "user", "content": prompt}],
                   **self.config["parameters"]}
        try:
            with httpx.Client(timeout=httpx.Timeout(90, connect=10), trust_env=False) as client:
                response = client.post(self.config["request_base_url"] + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.key}, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            entry["request_outcome"] = "not_sent"
            write_private(self.ledger, "".join(json.dumps(row) + "\n" for row in rows).encode())
            pending.unlink()
            raise LocalSummaryError("局域网摘要连接未建立；检查Mac与SSH隧道后续跑") from None
        except httpx.HTTPError as error:
            raise LocalSummaryError(f"局域网摘要提交状态未确认: {type(error).__name__}，pending保留") from None
        artifact = self.artifacts / (key + f"-{time.time_ns()}.http.json")
        save_http_response(artifact, response)
        if response.status_code != 200:
            if response.status_code in {400, 401, 403, 404, 422, 429}:
                entry["request_outcome"] = "rejected"
                entry["http_status"] = response.status_code
                write_private(self.ledger, "".join(json.dumps(row) + "\n" for row in rows).encode())
                pending.unlink()
            raise LocalSummaryError(f"局域网摘要HTTP {response.status_code}，原始回包保留")
        body = response.json()
        usage = body.get("usage", {})
        if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ["prompt_tokens", "completion_tokens"]):
            raise LocalSummaryError("局域网摘要usage无效，保留提交标记")
        entry.update(input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
                     tokens=usage["prompt_tokens"] + usage["completion_tokens"], response_artifact=str(artifact))
        write_private(self.ledger, "".join(json.dumps(row) + "\n" for row in rows).encode())
        choice = body["choices"][0]
        text = summary_text(body, self.config["model"])
        write_json(completed, {"text": text, "usage": usage, "response_artifact": str(artifact),
                               "finish_reason": choice["finish_reason"], "truncated": choice["finish_reason"] == "length"})
        pending.unlink()
        return text
