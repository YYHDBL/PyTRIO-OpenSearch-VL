"""用固定的 Anthropic Messages 兼容裁判评分并汇总 Pass@1。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

from dotenv import load_dotenv
import httpx

from agent_baseline import ROOT, write_json
from service_io import save_http_response
from answer_extraction import evaluation_response


# 来自 OpenSearch-VL eval_with_gpt4o.py；仅更换调用接口与裁判模型。
JUDGE_PROMPT = (
    "You are an impartial judge evaluating whether a deep research report contains the correct answer.\n\n"
    "[Question]\n{question}\n\n"
    "[Correct Answer]\n{correct_answer}\n\n"
    "[Deep Research Report]\n{response}\n\n"
    "Task: Determine if the deep research report contains the correct answer anywhere in its content.\n\n"
    "Instructions:\n"
    "1. Read through the entire research report carefully\n"
    "2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)\n"
    "3. Check if the information in the report is consistent with the correct answer\n"
    '4. The answer does NOT need to be in a specific format or labeled as "final answer"\n'
    "5. Provide your reasoning\n"
    '6. Answer with "yes" if the report contains the correct answer, "no" if it doesn\'t or contradicts it\n\n'
    "Output format:\n"
    "correct: [yes/no]\n"
    "reasoning: [your explanation]"
)


def parse_judge_response(text: str) -> bool:
    matches = re.findall(r"^\s*correct:\s*(yes|no)\s*$", text, flags=re.I | re.M)
    if len(matches) != 1:
        raise ValueError("裁判输出未提供唯一的 correct: yes/no")
    return matches[0].casefold() == "yes"


def judge_record(record: dict, output: Path, config: dict) -> dict:
    path = output / (record["sample_id"] + ".json")
    response_hash = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    if path.exists():
        result = json.loads(path.read_text())
        if result["response_sha256"] != response_hash or result["judge_config_sha256"] != config["sha256"]:
            raise ValueError("已有评分不匹配当前回答或裁判配置")
        return result
    pending_dir, responses_dir = output / "pending", output / "responses"
    pending_dir.mkdir(exist_ok=True)
    responses_dir.mkdir(exist_ok=True)
    pending = pending_dir / (record["sample_id"] + ".json")
    if pending.exists():
        raise RuntimeError("先前裁判请求结果未确认，请先检查原始回包，避免重复计费")
    report_text = evaluation_response(record)
    result = {"sample_id": record["sample_id"], "benchmark": record["benchmark"],
              "judge_model": config["model"], "judge_config_sha256": config["sha256"],
              "response_sha256": response_hash, "termination": record["termination"], "evaluation_response": report_text}
    if not report_text:
        result.update(correct=False, raw_judge="", reason="no_model_text", usage={})
    else:
        payload = {"model": config["model"], "max_tokens": 1024, "temperature": 0,
                   "thinking": {"type": "disabled"},
                   "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                       question=record["question"], correct_answer=record["reference"], response=report_text[:8000])}]}
        request_id = record["sample_id"]
        for attempt in range(3):
            write_json(pending, {"sample_id": request_id, "response_sha256": response_hash,
                                 "judge_config_sha256": config["sha256"], "at": time.time()})
            with (output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps({"sample_id": request_id, "at": time.time(), "attempt": attempt}) + "\n")
            try:
                with httpx.Client(timeout=90, trust_env=False) as client:
                    response = client.post(config["base_url"].rstrip("/") + "/v1/messages",
                                           headers={"x-api-key": os.environ["JUDGE_API_KEY"],
                                                    "anthropic-version": "2023-06-01"}, json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                pending.unlink()
                if attempt == 2:
                    raise RuntimeError(f"裁判网络失败: {type(exc).__name__}") from None
                time.sleep(2 ** attempt)
                continue
            except httpx.HTTPError as exc:
                raise RuntimeError(f"裁判请求结果未确认: {type(exc).__name__}；提交标记已保留") from None
            save_http_response(responses_dir / f"{request_id}-{time.time_ns()}.json", response)
            if response.status_code == 429 and attempt < 2:
                pending.unlink()
                time.sleep(2 ** attempt)
                continue
            if response.status_code != 200:
                raise RuntimeError(f"裁判请求 HTTP {response.status_code}")
            body = response.json()
            raw = "\n".join(block.get("text", "") for block in body.get("content", []) if block.get("type") == "text")
            if body.get("stop_reason") != "end_turn":
                raise RuntimeError("裁判输出未正常结束")
            result.update(correct=parse_judge_response(raw), raw_judge=raw, usage=body.get("usage", {}),
                          response_model=body.get("model"))
            break
    write_json(path, result)
    pending.unlink(missing_ok=True)
    return result


def summarize(records: list[dict], scores: list[dict], expected: int) -> dict:
    groups = defaultdict(list)
    for score in scores:
        groups[score["benchmark"]].append(score)
    per_benchmark = {name: {"correct": sum(s["correct"] for s in values), "count": len(values),
                            "pass_at_1": sum(s["correct"] for s in values) / len(values)}
                     for name, values in sorted(groups.items())}
    correct = sum(score["correct"] for score in scores)
    return {"expected": expected, "generated": len(records), "scored": len(scores),
            "complete": len(records) == len(scores) == expected, "correct": correct,
            "pass_at_1": correct / len(scores) if scores else None,
            "macro_pass_at_1": sum(v["pass_at_1"] for v in per_benchmark.values()) / len(per_benchmark) if per_benchmark else None,
            "per_benchmark": per_benchmark,
            "termination_counts": dict(Counter(r["termination"] for r in records)),
            "input_tokens": sum(t["input_tokens"] for r in records for t in r["trace"]),
            "output_tokens": sum(t["output_tokens"] for r in records for t in r["trace"]),
            "tool_calls": sum("action" in t for r in records for t in r["trace"]),
            "tool_errors": sum(bool(t.get("tool_error")) for r in records for t in r["trace"]),
            "judge": os.environ["JUDGE_MODEL"],
            "limitations": ["GLM judge replaces the paper's GPT-4o judge", f"custom {expected}-question subset",
                            "token counters do not replace actual account bills"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=200)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if os.environ.get("JUDGE_PROTOCOL") != "anthropic":
        raise ValueError("必须配置 Anthropic Messages 裁判协议")
    config = {"model": os.environ["JUDGE_MODEL"], "base_url": os.environ["JUDGE_BASE_URL"],
              "protocol": "anthropic", "temperature": 0, "max_tokens": 1024,
              "prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
              "answer_extraction_protocol": "OpenSearch-VL-1ac26e2-full-trajectory",
              "answer_extraction_sha256": hashlib.sha256((ROOT / "answer_extraction.py").read_bytes()).hexdigest()}
    config["sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("评分目录已有不同裁判配置")
    write_json(config_path, config)
    records = [json.loads(path.read_text()) for path in sorted(args.results_dir.glob("eval-*.json"))
               if not path.name.endswith(".partial.json")]
    scores = []
    for record in records:
        scores.append(judge_record(record, args.output_dir, config))
        print(json.dumps({"sample_id": record["sample_id"], "correct": scores[-1]["correct"]}), flush=True)
    summary = summarize(records, scores, args.expected)
    summary_tokens = args.results_dir / "summary-tokens.jsonl"
    if summary_tokens.exists():
        requests = [json.loads(line) for line in summary_tokens.open()]
        summary["summary_model"] = "Qwen/Qwen3.8-27B"
        summary["summary_model_input_tokens"] = sum(r.get("input_tokens", 0) for r in requests)
        summary["summary_model_output_tokens"] = sum(r.get("output_tokens", 0) for r in requests)
        summary["summary_model_unfinished_requests"] = sum("tokens" not in r for r in requests)
    bailian_usage = args.results_dir / "bailian-usage.jsonl"
    if bailian_usage.exists():
        rows = [json.loads(line) for line in bailian_usage.open()]
        summary["bailian_image_search"] = {
            "billed_requests": len(rows),
            "input_tokens": sum(r.get("input_tokens") or 0 for r in rows),
            "output_tokens": sum(r.get("output_tokens") or 0 for r in rows),
            "reported_image_search_calls": sum(r.get("image_search_calls") or 0 for r in rows),
            "observed_completed_calls": sum(r.get("observed_completed_calls") or 0 for r in rows),
            "note": "only live billed requests are recorded; cache hits are excluded"}
    deepseek_usage = args.results_dir / "deepseek-usage.jsonl"
    if deepseek_usage.exists():
        rows = [json.loads(line) for line in deepseek_usage.read_text().splitlines()]
        summary["deepseek_search_fetch"] = {
            "live_requests": len(rows),
            "billable_search_requests": sum(r["operation"] == "search" and r["billable"] for r in rows),
            "billable_fetch_requests": sum(r["operation"] == "fetch" and r["billable"] for r in rows),
            "note": "successful nonempty responses only; cache hits excluded; reconcile actual bills"}
    doubao_usage = args.results_dir / "doubao-usage.jsonl"
    if doubao_usage.exists():
        rows = [json.loads(line) for line in doubao_usage.read_text().splitlines()]
        summary["doubao_image_search"] = {
            "completed_live_requests": len(rows),
            "with_results": sum(r["returned_doc_count"] > 0 for r in rows),
            "empty_responses": sum(r["returned_doc_count"] == 0 for r in rows),
            "provider_nonzero_codes": sum(r.get("provider_error_code") not in (None, 0) for r in rows),
            "returned_documents": sum(r["returned_doc_count"] for r in rows),
            "request_ids": [r["request_id"] for r in rows],
            "note": "cache hits excluded; visual pricing and actual charges require account verification"}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
