"""仅用 CPU 验收真实 SFT 的图文长度、assistant mask 和右移对齐。"""

from __future__ import annotations

import argparse
import base64
from bisect import bisect_right
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from protocol import IMAGE_PAD, TEMPLATE_PATH, build_messages, build_sft_datum


def validate_arrays(tokens, supervised, targets, weights, input_length: int) -> None:
    expected_weights = supervised[1:]
    expected_targets = np.where(expected_weights != 0, tokens[1:], 0)
    if not (input_length == len(targets) == len(weights) == len(tokens) - 1):
        raise ValueError("图文输入、target 和 weight 长度未对齐")
    if not np.array_equal(weights, expected_weights):
        raise ValueError("训练 mask 与独立计算的 assistant 区间不一致")
    if not np.array_equal(targets, expected_targets):
        raise ValueError("target 未按一位右移或未屏蔽非监督位置")
    if not np.any(weights):
        raise ValueError("没有有效 assistant 监督 token")


def render_oracle(messages: list[dict]) -> tuple[str, list[tuple[int, int, str]]]:
    parts, spans = [], []
    length = 0
    for message in messages:
        role = message["role"]
        content = message["content"]
        if not isinstance(content, str):
            content = "".join("<|vision_start|><|image_pad|><|vision_end|>" if item["type"] == "image"
                              else item["text"] for item in content)
        prefix = "<|im_start|>user\n<tool_response>\n" if role == "tool" else f"<|im_start|>{role}\n"
        suffix = "\n</tool_response><|im_end|>\n" if role == "tool" else "<|im_end|>\n"
        start = length + len(prefix)
        end = start + len(content) + (len("<|im_end|>") if role == "assistant" else 0)
        spans.append((start, end, role))
        piece = prefix + content + suffix
        parts.append(piece)
        length += len(piece)
    return "".join(parts), spans


def audit_row(row: dict, tokenizer, processor, image_root: Path) -> dict:
    messages = build_messages(row, image_root)
    text, spans = render_oracle(messages)
    template_text = tokenizer.apply_chat_template(messages, chat_template=TEMPLATE_PATH.read_text(), tokenize=False,
                                                  add_generation_prompt=False, enable_thinking=True)
    if text != template_text:
        raise ValueError("独立渲染与训练模板的完整对话不一致")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [span[0] for span in spans]
    owner = []
    for start, end in encoded["offset_mapping"]:
        index = bisect_right(starts, start) - 1
        role = "control"
        if index >= 0:
            left, right, candidate = spans[index]
            if left <= start < end <= right:
                role = candidate
            elif candidate == "assistant" and start < right < end:
                raise ValueError("token 跨越 assistant 监督区间边界")
        owner.append(role)
    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    image_refs = [item["image"] for message in messages if isinstance(message["content"], list)
                  for item in message["content"] if item["type"] == "image"]
    visuals, dimensions = [], []
    for ref in image_refs:
        with Image.open(ref) as image:
            width, height = image.size
        count = processor.get_number_of_image_patches(height, width, images_kwargs={}) // int(processor.merge_size) ** 2
        visuals.append(count)
        dimensions.append({"width": width, "height": height, "tokens": count})
    tokens, weights, roles = [], [], []
    image_index = 0
    for token, role in zip(encoded["input_ids"], owner, strict=True):
        if token == image_id:
            count = visuals[image_index]
            image_index += 1
            tokens.extend([0] * count)
            weights.extend([0] * count)
            roles.extend(["image"] * count)
        else:
            tokens.append(token)
            weights.append(int(role == "assistant"))
            roles.append(role)
    if image_index != len(image_refs):
        raise ValueError("图片数与实际视觉占位符不一致")
    tokens, weights = np.asarray(tokens), np.asarray(weights)
    datum = build_sft_datum(row, tokenizer, processor, image_root)
    actual_targets = datum.loss_fn_inputs["target_tokens"].to_numpy()
    actual_weights = datum.loss_fn_inputs["weights"].to_numpy()
    validate_arrays(tokens, weights, actual_targets, actual_weights, len(datum.model_input))
    actual_input = np.concatenate([np.asarray(chunk.tokens) if hasattr(chunk, "tokens")
                                   else np.zeros(chunk.expected_tokens, dtype=np.int64)
                                   for chunk in datum.model_input.chunks])
    if not np.array_equal(actual_input, tokens[:-1]):
        raise ValueError("实际 ModelInput 的文本或图片顺序与独立计算不一致")
    role_counts = Counter()
    for role, weight in zip(roles[1:], actual_weights, strict=True):
        if role != "assistant" and weight != 0:
            raise ValueError(f"非 assistant 区间被监督: {role}")
        role_counts[role] += 1
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    expected_eos = sum(message["role"] == "assistant" for message in messages)
    supervised_eos = int(((tokens[1:] == eos_id) & (actual_weights == 1)).sum())
    if supervised_eos != expected_eos:
        raise ValueError("assistant 结束 token 未完整参与监督")
    return {"sample_id": row["sample_id"], "source_dataset": row["source_dataset"],
            "input_tokens": len(datum.model_input), "visual_tokens": sum(visuals),
            "supervised_tokens": int(actual_weights.sum()), "masked_tokens": int((actual_weights == 0).sum()),
            "image_count": len(image_refs), "images": dimensions, "roles": dict(role_counts),
            "assistant_eos": supervised_eos, "mask_and_shift_passed": True}


def compare_remote_lengths(tokenizer, processor, baseline_dir: Path, eval_path: Path) -> dict:
    from agent_baseline import LEGACY_SYSTEM, sample_prompt
    from agent_tools import model_image_bytes

    evaluation = {row["sample_id"]: row for row in map(json.loads, eval_path.open())}
    checked, mismatches = 0, []
    for path in sorted(baseline_dir.glob("eval-*.json")):
        if path.name.endswith(".partial.json"):
            continue
        record = json.loads(path.read_text())
        row = evaluation[record["sample_id"]]
        with Image.open(row["images"][0]) as image:
            width, height = image.size
            raw = model_image_bytes(image)
        turns = [["system", LEGACY_SYSTEM, None], ["user", f"Image img_1 ({width} x {height} pixels). {row['question']}\n",
                                            base64.b64encode(raw).decode("ascii")]]
        prompt = sample_prompt(tokenizer, turns)
        visual = processor.get_number_of_image_patches(height, width, images_kwargs={}) // int(processor.merge_size) ** 2
        local = sum(len(chunk.tokens) for chunk in prompt.chunks if hasattr(chunk, "tokens")) + visual
        remote = record["trace"][0]["input_tokens"]
        checked += 1
        if local != remote:
            mismatches.append({"sample_id": row["sample_id"], "local": local, "remote": remote})
    return {"checked_first_turns": checked, "mismatches": mismatches,
            "method": "reuse existing baseline input_tokens; no new remote requests"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/selected/sft.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/source/sft"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.max_seq_len < 1:
        raise ValueError("max-seq-len 必须为正")
    args.output_dir.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(args.model_dir, use_fast=False, local_files_only=True)
    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    results, errors = [], []
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            try:
                result = audit_row(row, tokenizer, processor, args.image_root)
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
            except (ValueError, IndexError, KeyError, OSError) as exc:
                errors.append({"sample_id": row["sample_id"], "error": str(exc), "type": type(exc).__name__})
            if (index + 1) % 32 == 0:
                print(f"audited={index + 1}/{len(rows)} passed={len(results)} errors={len(errors)}", flush=True)
    remote = compare_remote_lengths(tokenizer, processor, Path("outputs/base-v1"), Path("data/selected/eval.jsonl"))
    lengths = [item["input_tokens"] for item in results]
    over = [item["sample_id"] for item in results if item["input_tokens"] > args.max_seq_len]
    totals = {key: sum(item[key] for item in results)
              for key in ("input_tokens", "visual_tokens", "supervised_tokens", "masked_tokens", "image_count", "assistant_eos")}
    model_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.model_dir.iterdir()
                   if p.name in {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json", "config.json"}}
    summary = {"samples": len(rows), "passed_mask_and_shift": len(results), "errors": errors,
               "max_seq_len": args.max_seq_len, "overlength_count": len(over), "overlength_ids": over,
               "lengths": {"min": min(lengths), "mean": float(np.mean(lengths)),
                           "median": float(np.median(lengths)), "p90": float(np.quantile(lengths, .9)),
                           "p95": float(np.quantile(lengths, .95)), "max": max(lengths)} if lengths else {},
               "totals": totals, "remote_length_check": remote,
               "cpu_audit_passed": len(results) == len(rows) and not over,
               "remote_lengths_passed": remote["checked_first_turns"] > 0 and not remote["mismatches"],
               "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
               "template_sha256": hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest(),
               "protocol_sha256": hashlib.sha256(Path("protocol.py").read_bytes()).hexdigest(),
               "model_files_sha256": model_files,
               "versions": {name: version(name) for name in ("transformers", "pytrio", "numpy", "pillow")},
               "processor": {"class": type(processor).__name__, "size": dict(processor.size),
                             "patch_size": processor.patch_size, "merge_size": processor.merge_size},
               "scope": "current SFT template and system prompts; no training, dataset mutation or new remote requests"}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["cpu_audit_passed"] or not summary["remote_lengths_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
