"""作者fatal-aware训练流程的PyTRIO适配；默认仅离线预检。"""

import argparse
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytrio as trio
from dotenv import load_dotenv
from transformers import AutoImageProcessor

from agent_baseline import ROOT, TokenBudget, author_system
from agent_tools import ToolEnvironment, ToolError
from judge_baseline import parse_judge_response
from rl_diagnostic import author_prompts, judge_request, parse_score, reward_parts, trajectory_summary
from rl_loss import make_ppo_loss
from rl_prepare import build_trajectory_datums, rloo_advantages
from rl_rollout import dump_prompt, load_prompt, run_rollout
from service_io import write_json
from train_sft import remote_operation


def validate_selection(selected, available, smoke):
    if not isinstance(selected, list) or not all(isinstance(k, str) for k in selected):
        raise ValueError("选题必须为字符串ID数组")
    if len(selected) != len(set(selected)) or not set(selected) <= set(available):
        raise ValueError("选题ID必须唯一且属于冻结RL128题")
    if (smoke and len(selected) != 2) or (not smoke and len(selected) < 4):
        raise ValueError("smoke限2题且仅一批，正式训练至少4题")


def count_optimizer_updates(operations):
    return sum(item["operation"] == "update" for item in operations)


def actor_metadata_from_response(response, meta):
    values = response["loss_fn_outputs"][0]["logprobs"]["data"]
    start = meta["start"]
    old = values[start:start + len(meta["old"])]
    if len(old) != len(meta["old"]) or not np.isfinite(old).all():
        raise ValueError("训练前Actor old logprob无效，停止更新")
    return {**meta, "sampling_old": list(meta["old"]), "old": old}


def score_rollout(record, row, judging, judge_config, prompts):
    parts = reward_parts(record)
    identifier = record["sample_id"]
    answer = parts["terminal_answer"]
    accuracy = int(judge_request(prompts[0].format(question=row["question"], reference_answer=row["answer"],
        assistant_answer=answer), identifier + "-accuracy", judging, judge_config, parse_judge_response)) if answer else 0
    query = judge_request(prompts[1].format(question=row["question"], ground_truth=row["answer"], prediction=answer,
        trajectory_summary=trajectory_summary(record["trace"][:parts["valid_steps"]])), identifier + "-query",
        judging, judge_config, parse_score) if parts["valid_steps"] else 0.0
    return {**parts, "accuracy": accuracy, "query": query,
            "reward": parts["format_reward"] * (0.8 * accuracy + 0.2 * query)}


def group_training_data(records, scores, aggregation):
    if len(records) != 4 or len(scores) != 4 or len({r["policy_id"] for r in records}) != 1:
        raise ValueError("训练需要来自同一策略的完整四条轨迹")
    fatal = [s["fatal_start"] is not None or r["termination"] != "final_answer" for r, s in zip(records, scores)]
    advantages = rloo_advantages([s["reward"] for s in scores], fatal)
    datums, metadata = [], []
    valid_trajectories = 0
    for record, score, advantage in zip(records, scores, advantages):
        steps = [{**s, "prompt": load_prompt(s["prompt"])} for s in record["trace"]]
        if not steps:
            continue
        cutoff = score["fatal_start"]
        built = build_trajectory_datums(steps, float(advantage), cutoff)
        if not built:
            continue
        valid_trajectories += 1
        kept = steps if cutoff is None else steps[:cutoff]
        denominator = sum(len(s["tokens"]) for s in kept) if aggregation == "seq-mean-token-mean" else 1
        for datum, step in zip(built, kept, strict=True):
            metadata.append({"start": len(step["prompt"]) - 1, "old": step["logprobs"],
                             "advantage": float(advantage), "normalizer": denominator})
            datums.append(trio.Datum(model_input=datum.model_input,
                                    loss_fn_inputs={"target_tokens": datum.loss_fn_inputs["target_tokens"]}))
    return datums, metadata, valid_trajectories


def save_pair(training, output, step, config, token_budget):
    state = remote_operation(output, "save-state", step,
        lambda: training.save_state(name=f"opensearch-rl-{step}-state").result().model_dump(mode="json"), token_budget=token_budget)
    sampler = remote_operation(output, "save-sampler", step,
        lambda: training.save_weights_for_sampler(name=f"opensearch-rl-{step}-sampler").result().model_dump(mode="json"), token_budget=token_budget)
    pair = {"step": step, "state_path": state["path"], "sampler_path": sampler["path"], "config_sha256": config["sha256"]}
    write_json(output / "latest-checkpoint.json", pair)
    write_json(output / "progress.json", {"step": step})
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-ids", type=Path, required=True, help="用户确认的RL题目ID JSON数组")
    parser.add_argument("--aggregation", choices=["seq-mean-token-sum", "seq-mean-token-mean"], required=True,
                        help="作者启动配置为sum，论文正文公式为mean；必须明确指定")
    parser.add_argument("--execute", action="store_true", help="显式允许远端采样和训练；默认只预检")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="独立2题、最多一次更新，不受整轮筛题完成限制")
    parser.add_argument("--token-budget", type=int, default=25_000_000)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    data_path = ROOT / "data/selected/rl_train.jsonl"
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != "fe38a4057291e98e633de4d8f917d04431f2baeb66c13c3a8344ccab86459af2":
        raise ValueError("冻结RL数据发生变化")
    all_rows = {r["sample_id"]: r for r in map(json.loads, data_path.read_text().splitlines())}
    selected = json.loads(args.selected_ids.read_text())
    validate_selection(selected, all_rows, args.smoke)
    if args.smoke:
        args.token_budget = min(args.token_budget, 1_000_000)
    rows = [all_rows[k] for k in selected]
    trained = json.loads((ROOT / "outputs/sft-2048-v1/sft_result.json").read_text())
    if not trained["complete"] or trained["steps"] != 512 or "sampler_weights" in trained["state_path"]:
        raise ValueError("必须从完整SFT State初始化RL")
    prompts = author_prompts(ROOT.parent / "OpenSearch-VL")
    system = author_system([json.loads(x) for x in (ROOT / "data/selected-seven-2048-v1/sft.jsonl").read_text().splitlines()])
    config = {"base_model": "Qwen/Qwen3.5-4B", "initial_state": trained["state_path"], "selected": selected,
              "aggregation": args.aggregation, "advantage": "author-rloo", "group_size": 4,
              "smoke": args.smoke,
              "questions_per_step": 4, "epochs": 1, "lr": 1e-6, "clip_low": 0.2, "clip_high": 0.28,
              "dual_clip": 3.0, "kl_loss": False, "temperature": 0.7, "context": 32768,
              "adam": {"learning_rate": 1e-6, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
                       "weight_decay": 0.01, "grad_clip_norm": 1.0},
              "judge": os.environ["JUDGE_MODEL"], "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
              "prompts_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
              "code_sha256": {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                              ["train_rl.py", "rl_loss.py", "rl_prepare.py", "rl_rollout.py", "rl_diagnostic.py", "agent_tools.py", "agent_baseline.py"]}}
    config["sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    inspect.signature(trio.TrainingClient.forward_backward_custom).bind(None, data=[], loss_fn=lambda *_: None)
    inspect.signature(trio.ServiceClient.create_training_client_from_state).bind(None, path=trained["state_path"])
    if not args.execute:
        print(json.dumps({"preflight": True, "network_requests": 0, "updates": 0,
                          "questions": len(rows), "planned_groups": len(rows), "aggregation": args.aggregation}))
        return
    diagnostic = json.loads((ROOT / "outputs/rl-diagnostic-128x4-v1/status.json").read_text())
    if not diagnostic["complete"] and not args.smoke:
        raise ValueError("筛题诊断尚未完整结束，禁止启动RL")
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
        os.environ.pop(k, None)
    os.environ["PADDLE_OCR_PROXY"] = ""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cp = args.output_dir / "config.json"
        if cp.exists() and json.loads(cp.read_text()) != config:
            raise ValueError("已有RL目录配置不一致")
        if cp.exists() and not args.resume:
            raise ValueError("已有训练目录必须显式resume")
        write_json(cp, config)
        (args.output_dir / "operations").mkdir(exist_ok=True)
        if (args.output_dir / "pending-operation.json").exists():
            raise ValueError("存在未确认训练操作，不自动重发")
        service = trio.ServiceClient(api_key=os.environ["PYTRIO_API_KEY"])
        cap = next(m for m in service.get_server_capabilities().supported_models if m.model_name == config["base_model"])
        if not cap.training.available or cap.training.max_seq_len < 32768:
            raise ValueError("训练服务当前不能接收32k样本")
        if args.resume:
            pair = json.loads((args.output_dir / "latest-checkpoint.json").read_text())
            updates = [v["step"] for f in (args.output_dir / "operations").glob("*.json")
                       if (v := json.loads(f.read_text()))["operation"] == "update"]
            if pair["config_sha256"] != config["sha256"] or max(updates, default=pair["step"]) > pair["step"]:
                raise ValueError("有更新超出checkpoint，不恢复旧权重重复训练")
            training = service.create_training_client_from_state_with_optimizer(pair["state_path"])
        else:
            training = service.create_training_client_from_state(trained["state_path"])
            pair = save_pair(training, args.output_dir, 0, config, args.token_budget)
        sampler = service.create_sampling_client(base_model=config["base_model"], model_path=pair["sampler_path"])
        tokenizer = sampler.get_tokenizer()
        processor = AutoImageProcessor.from_pretrained(tokenizer.name_or_path, use_fast=False)
        budget = TokenBudget(args.output_dir / "tokens.jsonl", args.token_budget)
        summary_budget = TokenBudget(args.output_dir / "summary-tokens.jsonl", 250_000 if args.smoke else 8_000_000)
        summary_sampler = service.create_sampling_client(base_model="Qwen/Qwen3.8-27B")
        summary_tokenizer = summary_sampler.get_tokenizer()
        current = {"id": ""}
        def summarize(prompt):
            ids = summary_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                return_dict=False, add_generation_prompt=True, enable_thinking=False)
            value = summary_budget.sample(summary_sampler, trio.ModelInput.from_ints(ids), current["id"],
                trio.SamplingParams(max_tokens=1024, temperature=0.3, top_p=0.95, top_k=20, seed=42)).sequences[0].text
            if not value.strip():
                raise ToolError("训练摘要为空")
            return value
        tools = ToolEnvironment(ROOT / "data/tool_cache/author-seven-v5", image_limit=800, summarizer=summarize,
            summary_model="Qwen/Qwen3.8-27B", ocr_cache_dir=ROOT / "data/tool_cache/ocr-v1", usage_dir=args.output_dir)
        judge_config = {"model": os.environ["JUDGE_MODEL"], "url": os.environ["JUDGE_BASE_URL"].rstrip("/") + "/v1/messages"}
        for offset in range(pair["step"] * 4, len(rows), 4):
            step = offset // 4 + 1
            directory = args.output_dir / f"batch-{step:04d}"
            directory.mkdir(exist_ok=True)
            judging = directory / "judges"
            judging.mkdir(exist_ok=True)
            datums, metadata, denominator = [], [], 0
            for question_index, row in enumerate(rows[offset:offset + 4], offset):
                records, scores = [], []
                for replica in range(4):
                    identifier = f"{row['sample_id']}-r{replica}"
                    current["id"] = f"step-{step}-{identifier}"
                    record = run_rollout(row, identifier, sampler, tokenizer, processor, tools, budget, directory,
                                         system, pair["sampler_path"], 42 + question_index * 4 + replica)
                    records.append(record)
                    score = score_rollout(record, row, judging, judge_config, prompts)
                    scores.append(score)
                    write_json(directory / f"{identifier}.score.json", score)
                ds, ms, count = group_training_data(records, scores, args.aggregation)
                datums.extend(ds)
                metadata.extend(ms)
                denominator += count
            if any(m["advantage"] != 0 for m in metadata):
                # 沿用作者更新前Actor重算old_log_probs，原始采样值仍保留用于审计。
                actor_metadata = []
                for datum, meta in zip(datums, metadata, strict=True):
                    response = remote_operation(args.output_dir, "old-actor", step,
                        lambda d=datum: training.forward([d], loss_fn="cross_entropy").result().model_dump(mode="json"),
                        input_tokens=len(datum.model_input), token_budget=args.token_budget)
                    actor_metadata.append(actor_metadata_from_response(response, meta))
                write_json(directory / "actor-old-logprobs.json", {"metadata": actor_metadata,
                           "policy_id": pair["sampler_path"], "computed_before_optimizer_update": True})
                metadata = actor_metadata
                for datum, meta in zip(datums, metadata, strict=True):
                    remote_operation(args.output_dir, "backward", step,
                        lambda d=datum, m=meta: training.forward_backward_custom([d],
                            make_ppo_loss([m], denominator)).result().model_dump(mode="json"),
                        input_tokens=2 * len(datum.model_input), token_budget=args.token_budget)
                remote_operation(args.output_dir, "update", step,
                    lambda: training.optim_step(trio.AdamParams(**config["adam"])).result().model_dump(mode="json"),
                    token_budget=args.token_budget)
                if args.smoke:
                    probe = datums[0]
                    reference = remote_operation(args.output_dir, "live-after-update", step,
                        lambda: training.forward([probe], loss_fn="cross_entropy").result().model_dump(mode="json"),
                        input_tokens=len(probe.model_input), token_budget=args.token_budget)
                    write_json(args.output_dir / "live-after-update.json", {
                        "model_input": dump_prompt(probe.model_input),
                        "targets": probe.loss_fn_inputs["target_tokens"].data,
                        "logprobs": reference["loss_fn_outputs"][0]["logprobs"]["data"]})
            pair = save_pair(training, args.output_dir, step, config, args.token_budget)
            sampler = service.create_sampling_client(base_model=config["base_model"], model_path=pair["sampler_path"])
            print(json.dumps({"step": step, "groups": len(rows[offset:offset + 4]), "datums": len(datums)}), flush=True)
        updates = count_optimizer_updates([json.loads(f.read_text()) for f in
                                          (args.output_dir / "operations").glob("*.json")])
        if args.smoke and updates != 1:
            raise RuntimeError("smoke未完成恰好一次参数更新，不能宣称通过")
        write_json(args.output_dir / "result.json", {"complete": True, "checkpoint": pair,
                                                    "optimizer_updates": updates, "smoke": args.smoke})


if __name__ == "__main__":
    main()
