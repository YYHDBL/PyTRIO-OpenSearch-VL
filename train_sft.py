"""从固定的公开 SFT 轨迹训练 Qwen3.5-4B LoRA。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytrio as trio
from dotenv import load_dotenv
from transformers import AutoImageProcessor

from protocol import build_sft_datum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weights-name", default="opensearch-vl-qwen35-4b-sft")
    args = parser.parse_args()
    if min(args.rank, args.epochs, args.batch_size, args.max_seq_len) < 1:
        parser.error("rank、epochs、batch-size 和 max-seq-len 必须大于 0")
    load_dotenv(Path(__file__).parent / ".env", override=False)

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 512:
        raise ValueError(f"当前方案要求 512 条 SFT 轨迹，数据文件有 {len(rows)} 条")

    client = trio.ServiceClient(api_key=os.environ.get("PYTRIO_API_KEY"))
    supported = client.get_supported_models()
    if args.base_model not in supported:
        raise ValueError(f"PyTRIO 账号暂不支持 {args.base_model}；当前支持：{supported}")
    training = client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank,
        seed=42,
    )
    tokenizer = training.get_tokenizer()
    processor_source = getattr(tokenizer, "name_or_path", args.base_model)
    image_processor = AutoImageProcessor.from_pretrained(processor_source, use_fast=False)

    datums = [build_sft_datum(row, tokenizer, image_processor, args.image_root) for row in rows]
    overlength = [index for index, datum in enumerate(datums) if len(datum.model_input) > args.max_seq_len]
    if overlength:
        raise ValueError(
            f"有 {len(overlength)} 条轨迹超过 max-seq-len={args.max_seq_len}；"
            "为保持固定的 512 条样本，不会静默截断或丢弃，请先调整实验配置。"
        )

    adam = trio.AdamParams(learning_rate=args.learning_rate)
    step = 0
    for epoch in range(args.epochs):
        order = np.random.default_rng(42 + epoch).permutation(len(datums))
        for start in range(0, len(order), args.batch_size):
            batch = [datums[int(index)] for index in order[start : start + args.batch_size]]
            forward = training.forward_backward(batch, loss_fn="cross_entropy")
            update = training.optim_step(adam)
            result = forward.result()
            update.result()
            weights = np.concatenate([datum.loss_fn_inputs["weights"].to_numpy() for datum in batch])
            logprobs = np.concatenate([item["logprobs"].to_numpy() for item in result.loss_fn_outputs])
            loss = -float(np.dot(logprobs, weights) / weights.sum())
            step += 1
            print(f"epoch={epoch + 1} step={step} loss={loss:.5f} supervised_tokens={weights.sum():.0f}", flush=True)

    sampler = training.save_weights_for_sampler(name=args.weights_name).result()
    state = training.save_state(name=f"{args.weights_name}-state").result()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "sft_result.json"
    result_path.write_text(
        json.dumps({
            "base_model": args.base_model,
            "rank": args.rank,
            "epochs": args.epochs,
            "steps": step,
            "samples": len(datums),
            "max_seq_len": args.max_seq_len,
            "sampler_path": sampler.path,
            "state_path": state.path,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"sampler_path={sampler.path}")
    print(f"state_path={state.path}")
    print(f"record={result_path}")


if __name__ == "__main__":
    main()
