"""固定图文轨迹的 LoRA SFT：先离线检查，再训练，成对保存并验证恢复。"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import numpy as np
import pytrio as trio
from dotenv import load_dotenv
from pytrio.types.forward.forward_backward_output import ForwardBackwardOutput
from transformers import AutoImageProcessor, AutoTokenizer

from agent_baseline import sample_prompt
from agent_tools import model_image_bytes
from PIL import Image
from protocol import TEMPLATE_PATH, build_sft_datum, system_with_tools
from service_io import write_json


ROOT = Path(__file__).resolve().parent


def checkpoint_schedule(steps: int, interval: int) -> list[int]:
    if steps < 8 or interval < 1:
        raise ValueError("训练至少八步，checkpoint 间隔必须为正")
    return sorted({8, steps, *range(interval, steps, interval)})


def batch_order(size: int, batch_size: int, epochs: int, seed: int) -> list[list[int]]:
    if min(size, batch_size, epochs) < 1:
        raise ValueError("样本数、批次大小与轮数必须为正")
    batches = []
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(size).tolist()
        batches.extend(order[i:i + batch_size] for i in range(0, size, batch_size))
    return batches


def loss_per_token(result: ForwardBackwardOutput, batch: list) -> float:
    if len(result.loss_fn_outputs) != len(batch):
        raise ValueError("服务返回的 loss 样本数与训练批次不一致")
    weights = np.concatenate([np.asarray(d.loss_fn_inputs["weights"].data) for d in batch])
    logprobs = np.concatenate([np.asarray(item["logprobs"].data) for item in result.loss_fn_outputs])
    if weights.shape != logprobs.shape or not weights.sum() > 0:
        raise ValueError("监督位置与 logprobs 长度或权重无效")
    active = weights > 0
    loss = -float(np.dot(logprobs[active], weights[active]) / weights.sum())
    if not np.isfinite(loss) or loss < 0:
        raise ValueError("训练 loss 非有限值或为负，停止更新")
    return loss


def resume_checkpoint(output: Path, config_hash: str) -> dict:
    if (output / "pending-operation.json").exists():
        raise RuntimeError("先前远程操作状态未确认，必须先核对原请求，禁止自动重复提交")
    pair = json.loads((output / "latest-checkpoint.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    if not pair.get("state_path") or not pair.get("sampler_path") or pair.get("config_sha256") != config_hash:
        raise ValueError("恢复需要匹配配置且两类权重都完整的 checkpoint")
    updates = [json.loads(p.read_text())["step"] for p in (output / "operations").glob("*-update-*.json")]
    if progress["step"] != pair["step"] or max(updates, default=pair["step"]) != pair["step"]:
        raise ValueError("已有确认更新超出 checkpoint，不能静默恢复旧权重后重复训练")
    return pair


def remote_operation(output: Path, operation: str, step: int, action, input_tokens: int = 0,
                     output_tokens: int = 0, token_budget: int = 6_000_000) -> dict:
    pending = output / "pending-operation.json"
    if pending.exists():
        raise RuntimeError("存在未确认操作，保留断点并停止，不能盲目重发")
    ledger = output / "requests.jsonl"
    requests = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    reservation = input_tokens + output_tokens
    if sum(r["reserved_tokens"] for r in requests) + reservation > token_budget:
        raise RuntimeError("达到 SFT 本次提交 token 保护上限")
    request_id = f"{time.time_ns()}-{operation}-{step:04d}"
    entry = {"request_id": request_id, "operation": operation, "step": step, "at": time.time(),
             "reserved_tokens": reservation, "input_tokens": input_tokens, "max_output_tokens": output_tokens}
    write_json(pending, entry)
    with ledger.open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    start = time.monotonic()
    result = action()
    write_json(output / "operations" / (request_id + ".json"),
               {**entry, "seconds": time.monotonic() - start, "result": result})
    pending.unlink()
    return result


def diagnostic_loss(training, batch: list, output: Path, step: int, token_budget: int) -> float:
    result = remote_operation(output, "forward", step,
                              lambda: training.forward(batch, loss_fn="cross_entropy").result().model_dump(mode="json"),
                              sum(len(d.model_input) for d in batch), token_budget=token_budget)
    return loss_per_token(ForwardBackwardOutput.model_validate(result), batch)


def save_pair(training, output: Path, step: int, config: dict, diagnostic: list, token_budget: int) -> dict:
    nll = diagnostic_loss(training, diagnostic, output, step, token_budget)
    prefix = f"{config['weights_name']}-step-{step:04d}"
    state = remote_operation(output, "save-state", step,
                             lambda: training.save_state(name=prefix + "-state").result().model_dump(mode="json"),
                             token_budget=token_budget)
    sampler = remote_operation(output, "save-sampler", step,
                               lambda: training.save_weights_for_sampler(name=prefix + "-sampler").result().model_dump(mode="json"),
                               token_budget=token_budget)
    pair = {"step": step, "config_sha256": config["sha256"], "state_path": state["path"],
            "sampler_path": sampler["path"], "diagnostic_loss": nll, "at": time.time()}
    write_json(output / f"checkpoint-{step:04d}.json", pair)
    write_json(output / "latest-checkpoint.json", pair)
    print(json.dumps({"checkpoint_step": step, "diagnostic_loss": nll, "both_weights_saved": True}), flush=True)
    return pair


def restore_training(service, pair: dict, diagnostic: list, output: Path, token_budget: int):
    training = service.create_training_client_from_state_with_optimizer(pair["state_path"])
    restored_nll = diagnostic_loss(training, diagnostic, output, pair["step"], token_budget)
    if not np.isclose(restored_nll, pair["diagnostic_loss"], rtol=1e-3, atol=1e-3):
        raise ValueError("State 恢复后的固定样本 loss 与保存前不一致")
    write_json(output / f"restore-verification-{pair['step']:04d}.json", {"step": pair["step"],
        "before_loss": pair["diagnostic_loss"], "restored_loss": restored_nll, "with_optimizer": True})
    return training


def record_pilot(output: Path, pair: dict, batch_size: int) -> None:
    health = json.loads((output / "sampler-health-0008.json").read_text())
    restored = json.loads((output / "restore-verification-0008.json").read_text())
    if pair["step"] != 8 or health["step"] != 8 or restored["step"] != 8 or not restored["with_optimizer"]:
        raise ValueError("前八步的 Sampler/State 恢复检查尚未全部通过")
    if min(health["input_tokens"], health["output_tokens"]) < 1:
        raise ValueError("Sampler 检查没有有效图文输入或输出")
    initial = json.loads((output / "initial-loss.json").read_text())["loss"]
    if np.isclose(initial, pair["diagnostic_loss"], rtol=0, atol=1e-6):
        raise ValueError("前八步固定样本 loss 没有可观测变化，先检查参数更新，不直接继续")
    write_json(output / "pilot-verification.json", {"steps": 8, "samples_seen": 8 * batch_size,
        "initial_loss": initial, "step8_loss": pair["diagnostic_loss"],
        "loss_delta": pair["diagnostic_loss"] - initial,
        "sampler_verified": True, "state_with_optimizer_verified": True,
        "continue_from_step": 9, "repeated_batches": 0})
    print(json.dumps({"pilot_verified": True, "step8_loss": pair["diagnostic_loss"],
                      "initial_loss": initial, "continue_from_step": 9}), flush=True)


def sampler_health(service, pair: dict, rows: list, tokenizer, processor, image_root: Path,
                   output: Path, token_budget: int, base_model: str) -> None:
    row = next(r for r in rows if len(r["images"]) == 1 and r["conversations"][0]["value"].count("<image>") == 1)
    with Image.open(image_root / row["images"][0]) as image:
        raw = model_image_bytes(image)
    turns = [["system", system_with_tools(row["system"], row["tools"]), None],
             ["user", row["conversations"][0]["value"], base64.b64encode(raw).decode("ascii")]]
    prompt = sample_prompt(tokenizer, turns, processor)
    sampler = service.create_sampling_client(base_model=base_model, model_path=pair["sampler_path"])
    result = remote_operation(output, "sampler-health", pair["step"],
                              lambda: sampler.sample(prompt=prompt, num_samples=1,
                                  sampling_params=trio.SamplingParams(max_tokens=128, temperature=0, seed=42,
                                                                    stop=["<|im_end|>"])).result().model_dump(mode="json"),
                              len(prompt), 128, token_budget)
    if result["input_tokens"] != len(prompt) or not result["sequences"][0]["tokens"]:
        raise ValueError("保存后的 sampler 权重未通过真实图文输入验证")
    write_json(output / f"sampler-health-{pair['step']:04d}.json",
               {"step": pair["step"], "training_sample_id": row["sample_id"],
                "input_tokens": result["input_tokens"], "output_tokens": len(result["sequences"][0]["tokens"]),
                "purpose": "training-data checkpoint loading check, not formal evaluation"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, default=ROOT / "outputs/sft-audit-seven-v1/summary.json")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--expected-samples", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weights-name", default="opensearch-vl-qwen35-4b-sft-v1")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--token-budget", type=int, default=6_000_000)
    args = parser.parse_args()
    if min(args.rank, args.epochs, args.batch_size, args.expected_samples, args.checkpoint_every, args.max_seq_len, args.token_budget) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("训练参数必须为正且有限")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


def run(args) -> None:
    output = args.output_dir
    (output / "operations").mkdir(exist_ok=True)
    audit = json.loads(args.audit_report.read_text())
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if not audit["cpu_audit_passed"] or audit["data_sha256"] != data_hash:
        raise ValueError("SFT 数据与已通过的 CPU 审计不一致")
    if hashlib.sha256((ROOT / "protocol.py").read_bytes()).hexdigest() != audit["protocol_sha256"] or hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest() != audit["template_sha256"]:
        raise ValueError("图文转换或模板已改变，必须重新审计")
    for name, expected in audit["model_files_sha256"].items():
        if hashlib.sha256((args.model_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"本地 tokenizer/processor 文件已改变：{name}")
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    if len(rows) != args.expected_samples or audit["samples"] != args.expected_samples:
        raise ValueError(f"数据和审计必须对应固定的 {args.expected_samples} 条轨迹")
    for row in rows:
        actual = {hashlib.sha256((args.image_root / ref).read_bytes()).hexdigest() for ref in row["images"]}
        if actual != set(row["image_sha256"]):
            raise ValueError(f"SFT 图片哈希变化：{row['sample_id']}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True, use_fast=False)
    datums = [build_sft_datum(row, tokenizer, processor, args.image_root) for row in rows]
    if max(len(d.model_input) for d in datums) > args.max_seq_len:
        raise ValueError("轨迹超出训练长度；禁止静默截断或丢弃")
    batches = batch_order(len(rows), args.batch_size, args.epochs, 42)
    if args.stop_after is not None and not 1 <= args.stop_after <= len(batches):
        raise ValueError("stop-after 超出预定更新步数")
    config = {"base_model": args.base_model, "rank": args.rank, "epochs": args.epochs,
              "batch_size": args.batch_size, "max_seq_len": args.max_seq_len, "learning_rate": args.learning_rate,
              "seed": 42, "train_mlp": True, "train_attn": True, "train_unembed": True,
              "data_sha256": data_hash, "samples": len(rows), "total_steps": len(batches),
              "weights_name": args.weights_name, "checkpoint_steps": checkpoint_schedule(len(batches), args.checkpoint_every),
              "data_order_sha256": hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol_sha256": audit["protocol_sha256"], "template_sha256": audit["template_sha256"],
              "pytrio_version": version("pytrio"), "loss_fn": "cross_entropy", "auto_shift": False,
              "gradient_normalization": "SDK logical-batch active-token totals; no extra manual scaling"}
    config["sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("输出目录已有不同训练配置")
    write_json(path, config)
    write_json(output / "cpu-preflight.json", {"samples": len(rows), "max_input_tokens": max(len(d.model_input) for d in datums),
        "input_tokens": sum(len(d.model_input) for d in datums), "total_steps": len(batches),
        "data_order": [[rows[i]["sample_id"] for i in batch] for batch in batches], "remote_training_created": False})
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "samples": len(rows), "steps": len(batches)}), flush=True)
        return
    if (output / "sft_result.json").exists() and json.loads((output / "sft_result.json").read_text()).get("complete"):
        print(json.dumps({"complete": True, "already_finished": True}), flush=True)
        return
    load_dotenv(ROOT / ".env")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(name, None)
    service = trio.ServiceClient(api_key=os.environ["PYTRIO_API_KEY"])
    cap = next(m for m in service.get_server_capabilities().supported_models if m.model_name == args.base_model)
    if not cap.training.available or args.max_seq_len > cap.training.max_seq_len or args.batch_size > cap.training.max_batch_size:
        raise ValueError("训练配置超出当前平台能力")
    if max(sum(len(datums[i].model_input) for i in b) for b in batches) > cap.training.max_tokens_per_request:
        raise ValueError("训练批次超出平台请求 token 容量")
    diagnostic = [datums[batches[0][0]]]
    if args.resume:
        pair = resume_checkpoint(output, config["sha256"])
        step = pair["step"]
        training = restore_training(service, pair, diagnostic, output, args.token_budget)
        if step == 8:
            sampler_health(service, pair, rows, tokenizer, processor, args.image_root, output, args.token_budget, args.base_model)
            record_pilot(output, pair, args.batch_size)
    else:
        if (output / "progress.json").exists() or (output / "pending-operation.json").exists():
            raise RuntimeError("已有训练记录，请核对后使用 resume，禁止重新开始覆盖")
        training = service.create_lora_training_client(base_model=args.base_model, rank=args.rank, seed=42,
                                                     train_mlp=True, train_attn=True, train_unembed=True)
        step = 0
        write_json(output / "progress.json", {"step": 0, "complete": False})
        initial_loss = diagnostic_loss(training, diagnostic, output, step, args.token_budget)
        write_json(output / "initial-loss.json", {"loss": initial_loss, "training_sample_id": rows[batches[0][0]]["sample_id"]})
    adam = trio.AdamParams(learning_rate=args.learning_rate)
    for index in range(step, len(batches)):
        ids = batches[index]
        batch = [datums[i] for i in ids]
        step = index + 1
        def update():
            forward = training.forward_backward(batch, loss_fn="cross_entropy")
            optim = training.optim_step(adam)
            return {"forward": forward.result().model_dump(mode="json"), "optimizer": optim.result().model_dump(mode="json")}
        start = time.monotonic()
        result = remote_operation(output, "update", step, update,
                                  sum(len(d.model_input) for d in batch), token_budget=args.token_budget)
        forward = ForwardBackwardOutput.model_validate(result["forward"])
        loss = loss_per_token(forward, batch)
        record = {"step": step, "loss": loss, "seconds": time.monotonic() - start,
                  "sample_ids": [rows[i]["sample_id"] for i in ids], "input_tokens": sum(len(d.model_input) for d in batch),
                  "supervised_tokens": sum(sum(d.loss_fn_inputs["weights"].data) for d in batch),
                  "learning_rate": args.learning_rate, "trainer_metrics": forward.metrics,
                  "optimizer_metrics": result["optimizer"]["metrics"]}
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        write_json(output / "progress.json", {"step": step, "complete": False})
        print(json.dumps(record, allow_nan=False), flush=True)
        if step in config["checkpoint_steps"] or step == args.stop_after:
            pair = save_pair(training, output, step, config, diagnostic, args.token_budget)
            if step == 8 or step == len(batches):
                sampler_health(service, pair, rows, tokenizer, processor, args.image_root, output, args.token_budget, args.base_model)
            if step == 8 and args.stop_after != 8:
                # 从刚保存的 State 继续第九步，前八步计入原定一轮，不重复训练。
                training = restore_training(service, pair, diagnostic, output, args.token_budget)
                record_pilot(output, pair, args.batch_size)
        if step == args.stop_after and step < len(batches):
            print(json.dumps({"stopped_at_checkpoint": step, "complete": step == len(batches)}), flush=True)
            return
    write_json(output / "sft_result.json", {**pair, "samples": len(rows), "steps": step, "complete": True,
                                             "base_model": args.base_model, "rank": args.rank})
    write_json(output / "progress.json", {"step": step, "complete": True})
    print(json.dumps({"complete": True, "steps": step, "both_weights_saved": True}), flush=True)


if __name__ == "__main__":
    main()
