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
    result = {"sample_id": record["sample_id"], "benchmark": record["benchmark"],
              "judge_model": config["model"], "judge_config_sha256": config["sha256"],
              "response_sha256": response_hash, "termination": record["termination"]}
    if not record["answer"]:
        result.update(correct=False, raw_judge="", reason="no_final_answer", usage={})
    else:
        payload = {"model": config["model"], "max_tokens": 1024, "temperature": 0,
                   "thinking": {"type": "disabled"},
                   "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                       question=record["question"], correct_answer=record["reference"], response=record["answer"])}]}
        request_id = record["sample_id"]
        for attempt in range(3):
            with (output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps({"sample_id": request_id, "at": time.time(), "attempt": attempt}) + "\n")
            try:
                with httpx.Client(timeout=90, trust_env=False) as client:
                    response = client.post(config["base_url"].rstrip("/") + "/v1/messages",
                                           headers={"x-api-key": os.environ["JUDGE_API_KEY"],
                                                    "anthropic-version": "2023-06-01"}, json=payload)
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise RuntimeError(f"裁判网络失败: {type(exc).__name__}") from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
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
            "limitations": ["GLM judge replaces the paper's GPT-4o judge", "custom 200-question subset",
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
              "prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()}
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
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
