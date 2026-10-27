"""从固定版本的公开数据构造评测、SFT 和 RL 子集。"""

from __future__ import annotations

import base64
import collections
import hashlib
import io
import json
import random
import re
import unicodedata
from pathlib import Path

import ijson
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/source"
BENCH = ROOT / "data/benchmarks/vision-deepresearch-eval/data"
OUTPUT = ROOT / "data/selected"
SEED = 42
REVISION = "deeaf45779a3bbd407d8f0ccb9b4831fc78e81c9"
EVAL_SOURCES = {
    "SimpleVQA": [("simplevqa_300.parquet", 40)],
    "InfoSeek": [("infoseek_300.parquet", 40)],
    "FVQA": [("fvqa_300.parquet", 20)],
    "VDR": [("testmini500_v1.parquet", 40)],
    "MMSearch": [("mmsearch_vision_171.parquet", 20)],
    "LiveVQA": [("livevqa_webwatcher_subset_300.parquet", 20)],
    "BrowseComp-VL": [("bcvl_level1_199.parquet", 6), ("bcvl_level2_200.parquet", 14)],
}
SFT_FILES = {
    "fvqa": "fvqa_llama_factory_clean.json",
    "livevqa": "livevqa_llama_factory_filtered.json",
    "palace": "palace_llama_factory_filtered.json",
    "webqa": "webqa_llama_factory_filtered.json",
    "wiki_art": "wikiart_llama_factory_filtered.json",
    "wiki_en": "wiki_en_llama_factory_filtered.json",
    "wiki_zh": "wiki_zh_llama_factory_filtered.json",
}
CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
IMAGE_PREFIX = re.compile(r"^image_id:\s*.*?\s+Question:\s*", re.S)
BAD_ANSWER = re.compile(
    r"\b(?:let me|i need to|i should|i will|i cannot|i can't|unable to|cannot determine|not enough information)\b|无法确定|无法判断|让我|我需要",
    re.I,
)
ALLOWED_TOOLS = {"crop", "image_search", "text_search"}


def key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).replace("<image>", " ").casefold().split())


def image_hash(raw: bytes) -> str:
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    return hashlib.sha256(raw).hexdigest()


def safe_image_path(root: Path, ref: str) -> Path:
    path = (root / ref).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"图片路径超出数据集目录: {ref}")
    return path


def read_image_paths(root: Path, refs: list[str]) -> tuple[list[str], set[str]]:
    paths, digests = [], set()
    for ref in refs:
        path = safe_image_path(root, ref)
        raw = path.read_bytes()
        paths.append(str(path))
        digests.add(image_hash(raw))
    return paths, digests


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_data(rng: random.Random) -> tuple[list[dict], set[str], set[str]]:
    rows, questions, images = [], set(), set()
    for dataset, files in EVAL_SOURCES.items():
        for filename, required in files:
            parquet = pq.ParquetFile(BENCH / filename)
            order = list(range(parquet.metadata.num_rows))
            rng.shuffle(order)
            # 先固定随机顺序，再按顺序读取候选，避免把完整图片列装入内存。
            chosen = {i: rank for rank, i in enumerate(order)}
            candidates = []
            index = 0
            for batch in parquet.iter_batches(batch_size=16, columns=["id", "question", "answer", "images", "image_packed"]):
                for row in batch.to_pylist():
                    candidates.append((chosen[index], index, row))
                    index += 1
            candidates.sort(key=lambda item: item[0])
            count = 0
            for _, index, row in candidates:
                question = IMAGE_PREFIX.sub("", row["question"]).strip()
                answer = str(row["answer"] or "").strip()
                qkey = key(question)
                if not qkey or not answer or qkey in questions:
                    continue
                packed = json.loads(row["image_packed"])
                if not packed or len(packed) != len(row["images"]):
                    continue
                blobs = [base64.b64decode(item["data"], validate=True) for item in packed]
                hashes = [image_hash(blob) for blob in blobs]
                if images.intersection(hashes):
                    continue
                saved = []
                for blob, digest in zip(blobs, hashes, strict=True):
                    with Image.open(io.BytesIO(blob)) as image:
                        suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format)
                    if suffix is None:
                        raise ValueError(f"不支持的评测图片格式: {dataset}/{row['id']}")
                    path = OUTPUT / "eval_images" / (digest + suffix)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(blob)
                    saved.append(str(path))
                rows.append({"sample_id": f"eval-{len(rows):04d}", "benchmark": dataset,
                             "source_file": filename, "source_revision": REVISION,
                             "source_row": index, "source_id": str(row["id"]),
                             "question": question, "answer": answer, "images": saved,
                             "image_sha256": hashes})
                questions.add(qkey)
                images.update(hashes)
                count += 1
                if count == required:
                    break
            if count != required:
                raise RuntimeError(f"{filename} 仅筛出 {count}/{required} 题")
            print(dataset, filename, count, flush=True)
    return rows, questions, images


def rl_data(rng: random.Random, questions: set[str], images: set[str]) -> list[dict]:
    groups: dict[str, list[tuple[int, dict]]] = collections.defaultdict(list)
    eval_questions = questions.copy()
    rl_questions: set[str] = set()
    with (SOURCE / "rl/rl_data.jsonl").open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            row = json.loads(raw)
            answer = row.get("answer", "").strip()
            if not row.get("images") or len(key(row.get("question", ""))) < 12:
                continue
            if not 1 <= len(answer) <= 500 or BAD_ANSWER.search(answer):
                continue
            groups[row["dataset"]].append((line, row))
    names = sorted(groups)
    if len(names) != 8 or min(map(len, groups.values())) < 5:
        raise RuntimeError("RL 来源或有效样本不足，无法按设定分配")
    # 每个来源保留至少 5 条，其余按合格候选数量分配。
    quotas = {name: 5 for name in names}
    for _ in range(128 - sum(quotas.values())):
        name = max(names, key=lambda n: (len(groups[n]) / (quotas[n] + 1), n))
        quotas[name] += 1
    selected = []
    for name in names:
        rng.shuffle(groups[name])
        count = 0
        for line, row in groups[name]:
            qkey = key(row["question"])
            if qkey in eval_questions:
                continue
            try:
                paths, hashes = read_image_paths(SOURCE / "rl", row["images"])
            except (OSError, ValueError, SyntaxError):
                continue
            if images.intersection(hashes):
                continue
            selected.append({**row, "sample_id": f"rl-{len(selected):04d}", "source_line": line,
                             "images": paths, "image_sha256": sorted(hashes)})
            rl_questions.add(qkey)
            images.update(hashes)
            count += 1
            if count == quotas[name]:
                break
        if count != quotas[name]:
            raise RuntimeError(f"RL {name} 仅筛出 {count}/{quotas[name]} 题")
        print("RL", name, count, flush=True)
    questions.update(rl_questions)
    return selected


def sft_data(rng: random.Random, questions: set[str], images: set[str]) -> list[dict]:
    selected = []
    blocked_questions = questions.copy()
    sft_questions: set[str] = set()
    for source_index, (name, filename) in enumerate(SFT_FILES.items()):
        path = SOURCE / "sft" / name / filename
        candidates = []
        with path.open("rb") as stream:
            for index, row in enumerate(ijson.items(stream, "item")):
                turns = row.get("conversations", [])
                refs = row.get("images", [])
                if not turns or not refs:
                    continue
                if sum(turn.get("value", "").count("<image>") for turn in turns) != len(refs):
                    continue
                question = next((t.get("value", "") for t in turns if t.get("from") == "human"), "")
                if not key(question) or sum(len(t.get("value", "")) for t in turns) > 32000:
                    continue
                calls = [m for t in turns if t.get("from") == "gpt" for m in CALL.findall(t.get("value", ""))]
                if not calls:
                    continue
                try:
                    tools = {json.loads(call)["name"] for call in calls}
                except (ValueError, KeyError, TypeError):
                    continue
                if not tools <= ALLOWED_TOOLS:
                    continue
                candidates.append((index, question, refs))
        rng.shuffle(candidates)
        required = 74 if source_index == 0 else 73
        accepted: dict[int, tuple[list[str], list[str]]] = {}
        for index, question, refs in candidates:
            qkey = key(question)
            if qkey in blocked_questions:
                continue
            try:
                paths, hashes = read_image_paths(path.parent, refs)
            except (OSError, ValueError, SyntaxError):
                continue
            if images.intersection(hashes):
                continue
            accepted[index] = (paths, sorted(hashes))
            sft_questions.add(qkey)
            images.update(hashes)
            if len(accepted) == required:
                break
        if len(accepted) != required:
            raise RuntimeError(f"SFT {name} 仅筛出 {len(accepted)}/{required} 条")
        with path.open("rb") as stream:
            for index, row in enumerate(ijson.items(stream, "item")):
                if index in accepted:
                    paths, hashes = accepted[index]
                    selected.append({**row, "sample_id": f"sft-{len(selected):04d}",
                                     "source_dataset": name, "source_row": index,
                                     "images": paths, "image_sha256": hashes})
        print("SFT", name, len(accepted), "candidates", len(candidates), flush=True)
    rng.shuffle(selected)
    questions.update(sft_questions)
    return selected


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"输出目录已存在，避免覆盖: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    rng = random.Random(SEED)
    evaluation, questions, images = eval_data(rng)
    rl = rl_data(rng, questions, images)
    sft = sft_data(rng, questions, images)
    if (len(evaluation), len(sft), len(rl)) != (200, 512, 128):
        raise RuntimeError("生成数量不符合实验设定")
    for name, rows in (("eval.jsonl", evaluation), ("sft.jsonl", sft), ("rl_train.jsonl", rl)):
        write_jsonl(OUTPUT / name, rows)
    manifest = {"seed": SEED, "eval_source": "Osilly/Vision-DeepResearch-Eval",
                "eval_revision": REVISION, "eval_counts": dict(collections.Counter(r["benchmark"] for r in evaluation)),
                "sft_source": "OpenSearch-VL/Search-VL-SFT-36K",
                "sft_revision": "2c1c460af4fa15bd63210cbf426a96664b959944",
                "rl_source": "OpenSearch-VL/Search-VL-RL-8K",
                "rl_revision": "8ef567289043eef004b13da83b0e7bb7f5ae2daa",
                "sft_counts": dict(collections.Counter(r["source_dataset"] for r in sft)),
                "rl_counts": dict(collections.Counter(r["dataset"] for r in rl)),
                "rules": {"overlap": "normalized question across splits and SHA-256 image bytes across and within splits; same-split questions may repeat with different images",
                          "sft": "74 FVQA and 73 per other source; only crop/image_search/text_search; <=32000 conversation characters; all image placeholders valid",
                          "rl": "answer 1-500 characters; exclude obvious incomplete thought phrases; minimum 5 per source, remainder weighted by eligible counts",
                          "evaluation": "VDR authors' released subsets; 6/14 BrowseComp-VL level 1/2; no image_caption fed to model"},
                "limitations": ["text character limit is not the exact multimodal token count",
                                "automated answer filtering does not establish factual correctness",
                                "FVQA dataset provenance must be checked against the cited 2017 benchmark",
                                "hashes do not catch resized or recompressed copies"]}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
