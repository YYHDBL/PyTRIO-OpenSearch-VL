"""从 OpenSearch-VL 已解压的数据中固定 SFT、RL 和评测子集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from PIL import Image

TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
WHITESPACE = re.compile(r"\s+")


def question_key(text: str) -> str:
    """归一化题面，避免相同问题跨训练与评测切分。"""
    text = unicodedata.normalize("NFKC", text).replace("<image>", " ")
    return WHITESPACE.sub(" ", text).strip().casefold()


def image_path(ref: str, root: Path) -> Path:
    path = (root / ref).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"图片路径越出数据目录：{ref}")
    return path


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sft(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"SFT JSON 顶层应为数组：{path}")

    selected = []
    for row in rows:
        turns = row.get("conversations", [])
        images = row.get("images", [])
        if not turns or not images or not any(
            turn.get("from") == "gpt" and TOOL_CALL.search(turn.get("value", ""))
            for turn in turns
        ):
            continue
        user = next((turn.get("value", "") for turn in turns if turn.get("from") == "human"), "")
        if not question_key(user):
            continue
        selected.append({"row": row, "question": question_key(user), "image_ref": images[0]})
    return selected


def read_rl(path: Path) -> list[dict]:
    selected = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            images = row.get("images", [])
            question = question_key(row.get("question", ""))
            answer = row.get("answer", "").strip()
            if not question or not answer or not images:
                continue
            selected.append({"row": row, "question": question, "image_ref": images[0], "line": line_number})
    return selected


def select_distinct(
    rows: list[dict],
    count: int,
    image_root: Path,
    blocked_questions: set[str],
    blocked_images: set[str],
) -> list[dict]:
    result = []
    questions = set(blocked_questions)
    images = set(blocked_images)
    for item in rows:
        if item["question"] in questions or len(result) == count:
            continue
        image = image_path(item["image_ref"], image_root)
        if not image.is_file():
            continue
        try:
            with Image.open(image) as opened:
                opened.verify()
        except (OSError, ValueError):
            continue
        digest = image_digest(image)
        if digest in images:
            continue
        questions.add(item["question"])
        images.add(digest)
        result.append({**item, "image": image, "image_hash": digest})
    if len(result) != count:
        raise RuntimeError(f"去重并验证图片后只有 {len(result)} 条，需要 {count} 条")
    return result


def write_jsonl(path: Path, rows: list[dict], split: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index, item in enumerate(rows):
            row = item["row"].copy()
            row["sample_id"] = f"{split}-{index:04d}"
            row["images"] = [str(item["image"])]
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-json", type=Path, required=True)
    parser.add_argument("--sft-image-root", type=Path, required=True)
    parser.add_argument("--rl-jsonl", type=Path, required=True)
    parser.add_argument("--rl-image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sft-count", type=int, default=512)
    parser.add_argument("--rl-count", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.sft_count, args.rl_count, args.eval_count) < 1:
        parser.error("每个子集数量都必须大于 0")

    rng = random.Random(args.seed)
    sft = read_sft(args.sft_json)
    rng.shuffle(sft)
    sft = select_distinct(sft, args.sft_count, args.sft_image_root, set(), set())

    sft_questions = {item["question"] for item in sft}
    sft_images = {item["image_hash"] for item in sft}
    rl = read_rl(args.rl_jsonl)
    rng.shuffle(rl)
    train = select_distinct(rl, args.rl_count, args.rl_image_root, sft_questions, sft_images)
    eval_rows = select_distinct(
        rl,
        args.eval_count,
        args.rl_image_root,
        sft_questions | {item["question"] for item in train},
        sft_images | {item["image_hash"] for item in train},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "sft.jsonl", sft, "sft")
    write_jsonl(args.output_dir / "rl_train.jsonl", train, "rl")
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows, "eval")
    manifest = {
        "seed": args.seed,
        "sft_count": len(sft),
        "rl_count": len(train),
        "eval_count": len(eval_rows),
        "sft_source": str(args.sft_json),
        "rl_source": str(args.rl_jsonl),
        "image_validation": "PIL verify + SHA-256 cross-split deduplication",
        "split_validation": "normalized question + image SHA-256 disjoint",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
