"""固定SFT模型对128道训练题各采样4次；只诊断，不更新权重。"""

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

import httpx
import pytrio as trio
from dotenv import load_dotenv
from transformers import AutoImageProcessor

from agent_baseline import ROOT, TokenBudget, author_system, run_question
from agent_tools import BudgetExhausted, ToolEnvironment, ToolError
from judge_baseline import parse_judge_response
from service_io import save_http_response, write_json


def reward_parts(record):
    steps = record["trace"]
    start = None
    streak = 0
    for index, step in enumerate(steps):
        streak = streak + 1 if step.get("tool_error") else 0
        if streak == 3:
            start = index - 2
            break
    valid = steps if start is None else steps[:start]
    pattern = r"^<think>.*?</think>\s*(<tool_call>.*?</tool_call>|<response>.*?</response>|<answer>.*?</answer>)\s*$"
    fmt = [int(not t.get("tool_error") and re.match(pattern, t["assistant"], re.S) is not None) for t in valid]
    answer = ""
    if record["termination"] == "final_answer" and start is None and steps:
        match = re.search(r"<(response|answer)>(.*?)</\1>", steps[-1]["assistant"], re.S)
        answer = match.group(2).strip() if match else ""
    return {"valid_steps": len(valid), "fatal_start": start,
            "format_reward": statistics.mean(fmt) if fmt else 0.0, "terminal_answer": answer}


def group_summary(scores):
    return {"complete": len(scores) == 4, "scored": len(scores),
            "correct_count": sum(s["accuracy"] for s in scores),
            "equal_rewards": len(scores) == 4 and len({s["reward"] for s in scores}) == 1,
            "rewards": [s["reward"] for s in scores]}


def trajectory_summary(steps):
    def truncate(text, limit):
        return text if len(text) <= limit else text[:limit // 2] + " ... " + text[-limit // 2:]
    lines = []
    calls = 0
    for i, step in enumerate(steps):
        if "action" in step:
            action = step["action"]
            args = truncate(json.dumps(action["arguments"], ensure_ascii=False), 200)
            status = " [ERROR]" if step.get("tool_error") else ""
            lines.append(f"Step {i + 1}: {action['name']}({args}){status}\n  Result: {truncate(step.get('observation', '').strip(), 400)}")
            calls += 1
            if calls >= 20:
                lines.append(f"... ({len(steps) - i - 1} more steps omitted)")
                break
        else:
            match = re.search(r"<(response|answer)>(.*?)</\1>", step["assistant"], re.S)
            if match:
                lines.append(f"Step {i + 1}: final_answer\n  Answer: {truncate(match.group(2).strip(), 300)}")
    return "\n\n".join(lines) or "(No tool calls in trajectory)"


def judge_request(prompt, identifier, root, config, parser):
    digest = hashlib.sha256(json.dumps({"prompt": prompt, "config": config}, sort_keys=True).encode()).hexdigest()
    result_path, pending = root / f"{identifier}.json", root / f"{identifier}.pending.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["sha256"] != digest:
            raise ValueError("诊断裁判缓存与当前配置不符")
        return result["value"]
    if pending.exists():
        raise ToolError("诊断裁判已有未确认提交，保留记录")
    payload = {"model": config["model"], "temperature": 0, "max_tokens": 1024,
               "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": prompt}]}
    for attempt in range(3):
        ledger = root / "requests.jsonl"
        if ledger.exists() and len(ledger.read_text().splitlines()) >= 1536:
            raise BudgetExhausted("达到本次诊断裁判1536次请求尝试上限")
        write_json(pending, {"at": time.time(), "sha256": digest})
        with (root / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps({"id": identifier, "attempt": attempt, "at": time.time()}) + "\n")
        try:
            with httpx.Client(timeout=90, trust_env=False) as client:
                response = client.post(config["url"], headers={"x-api-key": os.environ["JUDGE_API_KEY"],
                    "anthropic-version": "2023-06-01"}, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pending.unlink()
            if attempt == 2:
                raise ToolError("诊断裁判连接未建立") from None
            time.sleep(2 ** attempt)
            continue
        except httpx.HTTPError:
            raise ToolError("诊断裁判请求状态未知，pending保留") from None
        save_http_response(root / f"{identifier}-{time.time_ns()}.http.json", response)
        if response.status_code == 429:
            pending.unlink()
            if attempt == 2:
                raise ToolError("诊断裁判限流，当前轨迹评分挂起")
            time.sleep(30 * (attempt + 1))
            continue
        if response.status_code != 200:
            raise ToolError(f"诊断裁判HTTP {response.status_code}，原始记录保留")
        body = response.json()
        text = "\n".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        if body.get("stop_reason") != "end_turn":
            raise ToolError("诊断裁判未正常完成，原始记录保留")
        try:
            value = parser(text)
        except ValueError:
            raise ToolError("诊断裁判格式无效，原始记录保留") from None
        write_json(result_path, {"sha256": digest, "value": value, "text": text, "usage": body.get("usage", {})})
        pending.unlink()
        return value
    raise RuntimeError("诊断裁判没有结果")


def parse_score(text):
    matches = re.findall(r"^\s*score:\s*(\d+(?:\.\d+)?)\s*$", text, re.M | re.I)
    if len(matches) != 1 or not 0 <= float(matches[0]) <= 1:
        raise ValueError("搜索质量分数无效")
    return float(matches[0])


def author_prompts(author_root):
    tree = ast.parse((author_root / "RL/rllm/vision_deepresearch_async_workflow/deepresearch_workflow.py").read_text())
    query = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_QUERY_UTILITY_PROMPT_TEMPLATE" for t in n.targets))
    tree = ast.parse((author_root / "RL/rllm/rllm/rewards/deepresearch_reward.py").read_text())
    node = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "judging_prompt" for t in n.targets))
    return ast.literal_eval(node.func.value), query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, default=ROOT.parent / "OpenSearch-VL")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)
    os.environ["PADDLE_OCR_PROXY"] = ""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = ROOT / "data/selected/rl_train.jsonl"
        if hashlib.sha256(data.read_bytes()).hexdigest() != "fe38a4057291e98e633de4d8f917d04431f2baeb66c13c3a8344ccab86459af2":
            raise ValueError("RL128题哈希变化")
        rows = [json.loads(x) for x in data.read_text().splitlines()]
        if len(rows) != 128 or len({r["sample_id"] for r in rows}) != 128:
            raise ValueError("RL题目数量或ID重复")
        trained = json.loads((ROOT / "outputs/sft-2048-v1/sft_result.json").read_text())
        if not trained["complete"] or trained["steps"] != 512 or "/sampler_weights/" not in trained["sampler_path"]:
            raise ValueError("最终SFT Sampler不完整")
        accuracy_prompt, query_prompt = author_prompts(args.author_root)
        system = author_system([json.loads(x) for x in (ROOT / "data/selected-seven-2048-v1/sft.jsonl").read_text().splitlines()])
        judge_config = {"model": os.environ["JUDGE_MODEL"], "url": os.environ["JUDGE_BASE_URL"].rstrip("/") + "/v1/messages"}
        config = {"model": "Qwen/Qwen3.5-4B", "model_path": trained["sampler_path"], "temperature": 0.7,
                  "top_p": 1.0, "seed": 42, "seed_rule": "42 + question_index * 4 + replica",
                  "max_context": 32768, "max_tokens": 32768, "max_turns": 50, "fatal_error_threshold": 3,
                  "diagnostic": True, "generation_stop_tokens": ["<|im_end|>"], "group_size": 4,
                  "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                  "system_sha256": hashlib.sha256(system.encode()).hexdigest(), "judge": judge_config,
                  "accuracy_prompt": accuracy_prompt, "query_prompt": query_prompt,
                  "code_sha256": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                  ["rl_diagnostic.py", "agent_baseline.py", "agent_tools.py", "doubao_image.py", "deepseek_search.py", "paddle_ocr.py", "service_io.py", "protocol.py"]}}
        config["sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        cp = args.output_dir / "config.json"
        if cp.exists() and json.loads(cp.read_text()) != config:
            raise ValueError("诊断目录配置不同")
        write_json(cp, config)
        if args.preflight_only:
            print(json.dumps({"preflight": True, "questions": 128, "rollouts": 512, "config_sha256": config["sha256"]}))
            return
        service = trio.ServiceClient(api_key=os.environ["PYTRIO_API_KEY"])
        capability = next(m for m in service.get_server_capabilities().supported_models if m.model_name == config["model"])
        if not capability.sampling.available or min(capability.sampling.max_seq_len,
                                                   capability.sampling.max_tokens_per_request) < 32768:
            raise ValueError("当前采样服务不支持32k")
        sampler = service.create_sampling_client(base_model=config["model"], model_path=config["model_path"])
        tokenizer = sampler.get_tokenizer()
        processor = AutoImageProcessor.from_pretrained(tokenizer.name_or_path, use_fast=False)
        summary_sampler = service.create_sampling_client(base_model="Qwen/Qwen3.8-27B")
        summary_tokenizer = summary_sampler.get_tokenizer()
        budget = TokenBudget(args.output_dir / "tokens.jsonl", 25_000_000)
        summary_budget = TokenBudget(args.output_dir / "summary-tokens.jsonl", 8_000_000)
        current = {"id": ""}
        def summarize(prompt):
            ids = summary_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=False)
            result = summary_budget.sample(summary_sampler, trio.ModelInput.from_ints(ids), current["id"],
                trio.SamplingParams(max_tokens=1024, temperature=0.3, top_p=0.95, top_k=20, seed=42))
            if not result.sequences[0].text.strip():
                raise ToolError("摘要为空")
            return result.sequences[0].text.strip()
        tools = ToolEnvironment(ROOT / "data/tool_cache/author-seven-v5", image_limit=800, summarizer=summarize,
            summary_model="Qwen/Qwen3.8-27B", ocr_cache_dir=ROOT / "data/tool_cache/ocr-v1", usage_dir=args.output_dir)
        generation = args.output_dir / "rollouts"
        judging = args.output_dir / "judges"
        generation.mkdir(exist_ok=True)
        judging.mkdir(exist_ok=True)
        status = {"complete": False, "groups": {}, "deferred": {}, "no_weight_updates": True}
        try:
            for index, row in enumerate(rows):
                scores = []
                for replica in range(4):
                    identifier = f"{row['sample_id']}-r{replica}"
                    current["id"] = identifier
                    item = {**row, "sample_id": identifier, "benchmark": row["dataset"], "source_id": row["sample_id"],
                            "sampling_seed": 42 + index * 4 + replica}
                    try:
                        record = run_question(item, sampler, tokenizer, processor, tools, budget, generation, config, system)
                        parts = reward_parts(record)
                        accuracy = int(judge_request(accuracy_prompt.format(question=row["question"],
                            reference_answer=row["answer"], assistant_answer=parts["terminal_answer"]),
                            identifier + "-accuracy", judging, judge_config, parse_judge_response)) if parts["terminal_answer"] else 0
                        query = judge_request(query_prompt.format(question=row["question"], ground_truth=row["answer"],
                            prediction=parts["terminal_answer"], trajectory_summary=trajectory_summary(record["trace"][:parts["valid_steps"]])),
                            identifier + "-query", judging, judge_config, parse_score) if parts["valid_steps"] else 0.0
                        score = {"id": identifier, "accuracy": accuracy, "query": query, **parts,
                                 "reward": parts["format_reward"] * (0.8 * accuracy + 0.2 * query),
                                 "termination": record["termination"]}
                        write_json(args.output_dir / f"{identifier}.score.json", score)
                        scores.append(score)
                        print(json.dumps({"id": identifier, "accuracy": accuracy, "reward": score["reward"],
                                          "termination": record["termination"]}), flush=True)
                    except ToolError as error:
                        status["deferred"][identifier] = str(error)
                        print(json.dumps({"deferred": identifier, "error": str(error)}, ensure_ascii=False), flush=True)
                    status["groups"][row["sample_id"]] = group_summary(scores)
                    write_json(args.output_dir / "status.json", status)
        except BudgetExhausted as error:
            status["budget_stop"] = str(error)
        status["complete"] = len(status["groups"]) == 128 and all(g["complete"] for g in status["groups"].values())
        write_json(args.output_dir / "status.json", status)
        if not status["complete"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
