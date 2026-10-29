"""核验独立smoke更新及State/Sampler重载；只执行前向，不追加更新。"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytrio as trio
from dotenv import load_dotenv

from agent_baseline import ROOT
from rl_prepare import build_datum
from rl_rollout import load_prompt
from service_io import write_json
from train_sft import remote_operation


def compare_logprobs(before, restored, sampled):
    arrays = [np.asarray(x, dtype=np.float64) for x in (before, restored, sampled)]
    if not arrays[0].size or any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError("权重核验logprob长度或数值无效")
    delta = float(np.max(np.abs(arrays[0] - arrays[1])))
    error = float(np.max(np.abs(arrays[1] - arrays[2])))
    return {"weight_effect_observed": delta > 0, "before_after_max_difference": delta,
            "state_sampler_max_difference": error, "state_sampler_match": error <= 0.05,
            "cross_backend_absolute_tolerance": 0.05, "tokens": len(arrays[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    result = json.loads((root / "result.json").read_text())
    if not result.get("smoke") or result.get("optimizer_updates") != 1 or not result["complete"]:
        raise ValueError("仅核验恰好一次更新且完整完成的独立smoke")
    output = root / "verification"
    output.mkdir(exist_ok=True)
    (output / "operations").mkdir(exist_ok=True)
    if (output / "summary.json").exists():
        print((output / "summary.json").read_text())
        if not json.loads((output / "summary.json").read_text())["passed"]:
            raise SystemExit(2)
        return
    if (output / "pending-operation.json").exists():
        raise RuntimeError("存在未确认核验请求，不自动重复")
    load_dotenv(ROOT / ".env")
    original = json.loads(next((root / "operations").glob("*-save-state-0000.json")).read_text())["result"]["path"]
    record_path = next(p for p in (root / "batch-0001").glob("rl-*.json")
                       if not p.name.endswith((".score.json", ".pending.json")))
    record = json.loads(record_path.read_text())
    step = record["trace"][0]
    prompt = load_prompt(step["prompt"])
    tokens = step["tokens"]
    datum = build_datum(prompt, tokens, step["logprobs"], 1.)
    start = len(prompt) - 1
    datum = trio.Datum(model_input=datum.model_input, loss_fn_inputs={
        "target_tokens": datum.loss_fn_inputs["target_tokens"],
        "weights": np.asarray([0.] * start + [1.] * len(tokens), dtype=np.float32)})
    service = trio.ServiceClient(api_key=os.environ["PYTRIO_API_KEY"])
    restore_report = None
    live_path = root / "live-after-update.json"
    if live_path.exists():
        live = json.loads(live_path.read_text())
        restore_cache = output / "state-live-restore.json"
        if restore_cache.exists():
            restore_report = json.loads(restore_cache.read_text())
        else:
            restored_client = service.create_training_client_from_state_with_optimizer(result["checkpoint"]["state_path"])
            live_datum = trio.Datum(model_input=load_prompt(live["model_input"]),
                loss_fn_inputs={"target_tokens": np.asarray(live["targets"], dtype=np.int64)})
            raw = remote_operation(output, "state-live-restore", 1,
                lambda: restored_client.forward([live_datum], loss_fn="cross_entropy").result().model_dump(mode="json"),
                input_tokens=len(live_datum.model_input), token_budget=200_000)
            expected = np.asarray(live["logprobs"])
            actual = np.asarray(raw["loss_fn_outputs"][0]["logprobs"]["data"])
            if expected.shape != actual.shape or not np.isfinite(actual).all():
                raise ValueError("重载State概率形状或数值无效")
            delta = float(np.max(np.abs(actual - expected)))
            restore_report = {"live_vs_restored_max_difference": delta,
                              "absolute_tolerance": 0.001, "passed": delta <= 0.001}
            write_json(restore_cache, restore_report)
    def forward(name, path, optimizer):
        cache = output / f"{name}.json"
        if cache.exists():
            return json.loads(cache.read_text())["logprobs"]
        training = (service.create_training_client_from_state_with_optimizer(path) if optimizer
                    else service.create_training_client_from_state(path))
        response = remote_operation(output, name, 1,
            lambda: training.forward([datum], loss_fn="cross_entropy").result().model_dump(mode="json"),
            input_tokens=len(datum.model_input), token_budget=200_000)
        values = response["loss_fn_outputs"][0]["logprobs"]["data"][start:start + len(tokens)]
        write_json(cache, {"logprobs": values})
        return values
    cache = output / "sampler.json"
    if cache.exists():
        health = json.loads(cache.read_text())
    else:
        sampler = service.create_sampling_client(base_model="Qwen/Qwen3.5-4B", model_path=result["checkpoint"]["sampler_path"])
        raw = remote_operation(output, "sampler-health", 1,
            lambda: sampler.sample(prompt=prompt, num_samples=1,
                sampling_params=trio.SamplingParams(max_tokens=32, temperature=1., top_p=1., seed=42)).result().model_dump(mode="json"),
            input_tokens=len(prompt), output_tokens=32, token_budget=200_000)
        health = raw["sequences"][0]
        if raw["input_tokens"] != len(prompt):
            raise ValueError("Sampler重载核验图文token计数不同")
        write_json(cache, health)
    tokens, sampled = health["tokens"], health["logprobs"]
    datum = build_datum(prompt, tokens, sampled, 1.)
    datum = trio.Datum(model_input=datum.model_input, loss_fn_inputs={
        "target_tokens": datum.loss_fn_inputs["target_tokens"],
        "weights": np.asarray([0.] * start + [1.] * len(tokens), dtype=np.float32)})
    before = forward("before-health", original, False)
    restored = forward("restored-health", result["checkpoint"]["state_path"], True)
    summary = compare_logprobs(before, restored, sampled)
    summary.update({"optimizer_updates": 1, "additional_updates": 0,
                    "state_with_optimizer_loaded": True, "sampler_loaded": True,
                    "state_save_restore": restore_report})
    summary["passed"] = (summary["state_sampler_match"] and summary["weight_effect_observed"]
                         and (restore_report is None or restore_report["passed"]))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary))
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
