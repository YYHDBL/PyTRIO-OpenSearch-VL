"""在冻结的评测和 RL 集之外，按七工具协议重选 SFT 轨迹。"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from agent_tools import SUPPORTED_TOOLS
from select_experiment_data import CALL, ROOT, SEED, image_hash, key, sft_data, write_jsonl


def main() -> None:
    source = ROOT / "data/selected"
    output = ROOT / "data/selected-seven-v1"
    if output.exists():
        raise FileExistsError(f"输出目录已存在，避免覆盖: {output}")
    evaluation = [json.loads(line) for line in (source / "eval.jsonl").open()]
    rl = [json.loads(line) for line in (source / "rl_train.jsonl").open()]
    if (len(evaluation), len(rl)) != (200, 128):
        raise ValueError("冻结评测集或 RL 数据数量不正确")
    questions, images = set(), set()
    for row in evaluation + rl:
        questions.add(key(row["question"]))
        actual = {image_hash(Path(ref).read_bytes()) for ref in row["images"]}
        if actual != set(row["image_sha256"]):
            raise ValueError(f"图片哈希变化: {row['sample_id']}")
        images.update(actual)
    selected = sft_data(random.Random(SEED), questions, images, allowed_tools=SUPPORTED_TOOLS)
    if len(selected) != 512:
        raise ValueError("SFT 数量不是 512")
    counts = Counter()
    for row in selected:
        for turn in row["conversations"]:
            if turn.get("from") == "gpt":
                for call in CALL.findall(turn.get("value", "")):
                    counts[json.loads(call)["name"]] += 1
    output.mkdir(parents=True)
    write_jsonl(output / "sft.jsonl", selected)
    manifest = {"seed": SEED, "sft_count": len(selected),
                "source_counts": dict(Counter(row["source_dataset"] for row in selected)),
                "tool_call_counts": dict(sorted(counts.items())), "allowed_tools": sorted(SUPPORTED_TOOLS),
                "eval_sha256": hashlib.sha256((source / "eval.jsonl").read_bytes()).hexdigest(),
                "rl_sha256": hashlib.sha256((source / "rl_train.jsonl").read_bytes()).hexdigest(),
                "rules": "fixed eval and RL excluded by normalized question and image SHA-256; original system and tools retained; <=32000 conversation characters",
                "limitations": ["exact multimodal token lengths and masks must be re-audited",
                                "byte hash does not detect recompressed copies"]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
