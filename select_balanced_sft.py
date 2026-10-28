"""从公开完整轨迹选择 2048 条，并在冻结前完成逐条图文 CPU 审计。"""

from collections import Counter, deque
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
import time

import ijson
import numpy as np
from transformers import AutoImageProcessor, AutoTokenizer

from agent_baseline import outside_thought
from agent_tools import SUPPORTED_TOOLS
from audit_sft import audit_row
from protocol import TEMPLATE_PATH, system_with_tools
from select_experiment_data import CALL, IMAGE_PREFIX, ROOT, SFT_FILES, key, read_image_paths, write_jsonl


FAMILY_TARGETS = {"retrieval": 1228, "perception": 615, "enhancement": 205}
SOURCE_TARGETS = {"wiki_en": 410, "wiki_zh": 410, "webqa": 410, "livevqa": 307,
                  "fvqa": 205, "wiki_art": 154, "palace": 152}


def normalized_question(text: str) -> str:
    return key(IMAGE_PREFIX.sub("", text.replace("<image>", " ").strip()))


def trajectory_features(row: dict) -> dict | None:
    turns, refs = row.get("conversations", []), row.get("images", [])
    if not turns or not refs or any(t.get("from") not in {"human", "gpt", "observation"} for t in turns):
        return None
    if sum(t.get("value", "").count("<image>") for t in turns) != len(refs):
        return None
    calls = []
    for turn in turns:
        if turn["from"] != "gpt":
            continue
        found = CALL.findall(outside_thought(turn["value"]))
        if len(found) > 1:
            return None
        try:
            calls.extend(json.loads(c) for c in found)
        except (ValueError, TypeError):
            return None
    if not calls or any(not isinstance(c, dict) or c.get("name") not in SUPPORTED_TOOLS
                        or not isinstance(c.get("arguments"), dict) for c in calls):
        return None
    canonical = [json.dumps(c, sort_keys=True) for c in calls]
    if max(Counter(canonical).values()) > 2:
        return None
    final = next((outside_thought(t["value"]) for t in reversed(turns) if t["from"] == "gpt"), "")
    if not re.search(r"<response>\s*\S.*?</response>", final, re.S):
        return None
    for call in calls:
        args = call["arguments"]
        if call["name"] in {"text_search", "web_search"}:
            query, count = args.get("q", args.get("query")), args.get("top_k", 5)
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 128 or type(count) is not int or not 1 <= count <= 5:
                return None
        if call["name"] == "super_resolution" and args.get("scale", 4) != 4:
            return None
    tools = {c["name"] for c in calls}
    family = ("enhancement" if tools & {"sharpen", "super_resolution", "perspective_correct"}
              else "perception" if tools & {"crop", "layout_parsing"} else "retrieval")
    return {"tools": tools, "calls": calls, "family": family,
            "rounds": len(calls), "round_bin": "short" if len(calls) <= 2 else "medium" if len(calls) <= 5 else "long"}


def main() -> None:
    output = ROOT / "data/selected-seven-2048-v1"
    audit_output = ROOT / "outputs/sft-audit-2048-v1"
    if output.exists() or audit_output.exists():
        raise FileExistsError("已有选择或审计目录，不覆盖既有数据")
    output.mkdir(parents=True)
    audit_output.mkdir(parents=True)
    started = time.time()
    held_paths = [ROOT / "data/selected/eval.jsonl", ROOT / "data/selected/rl_train.jsonl"]
    frozen = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in held_paths}
    if list(frozen.values()) != ["59629e3226f24a1ea5c2f2ea06be92e25489844d66fcec627347d7380a9119b3",
                                 "fe38a4057291e98e633de4d8f917d04431f2baeb66c13c3a8344ccab86459af2"]:
        raise ValueError("原冻结评测或 RL 文件变化")
    held = [json.loads(l) for p in held_paths for l in p.read_text().splitlines()]
    blocked_questions = {normalized_question(r["question"]) for r in held}
    blocked_images = {h for r in held for h in r["image_sha256"]}
    prior = json.loads((ROOT / "outputs/sft-audit-seven-v1/summary.json").read_text())
    model = Path("/data/jyh-yyh/.cache/modelscope/models/Qwen--Qwen3.5-4B/snapshots/master")
    model_hashes = {n: hashlib.sha256((model / n).read_bytes()).hexdigest() for n in prior["model_files_sha256"]}
    if model_hashes != prior["model_files_sha256"]:
        raise ValueError("tokenizer/processor 与已有真实输入验证不一致")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model, use_fast=False, local_files_only=True)
    original = json.loads((ROOT / "data/selected-seven-v1/sft.jsonl").open().readline())
    system = system_with_tools(original["system"], original["tools"])
    rng = random.Random(42)
    candidates, rejected = [], Counter()
    for source, filename in SFT_FILES.items():
        path = ROOT / "data/source/sft" / source / filename
        with path.open("rb") as stream:
            for index, row in enumerate(ijson.items(stream, "item")):
                features = trajectory_features(row)
                if features is None:
                    rejected["structure_completion_repetition_or_backend_compatibility"] += 1
                    continue
                question = next((t["value"] for t in row["conversations"] if t["from"] == "human"), "")
                if not normalized_question(question) or normalized_question(question) in blocked_questions:
                    rejected["empty_or_heldout_question"] += 1
                    continue
                if system_with_tools(row.get("system", ""), row.get("tools", [])) != system:
                    rejected["different_system_or_tools"] += 1
                    continue
                candidates.append({"row": row, "features": features, "source": source, "filename": filename,
                                   "source_row": index, "priority": rng.random(), "visited": False})
        print(json.dumps({"scanned_source": source, "candidates_so_far": len(candidates)}), flush=True)
    candidates.sort(key=lambda c: c["priority"])
    selected, audits, features_kept = [], {}, {}
    sources, families, rounds, tool_presence = Counter(), Counter(), Counter(), Counter()
    selected_images = set()

    def accept(c):
        c["visited"] = True
        f, row = c["features"], c["row"]
        try:
            paths, hashes = read_image_paths(ROOT / "data/source/sft" / c["source"], row["images"])
            if hashes & (blocked_images | selected_images):
                rejected["heldout_or_repeated_image_bytes"] += 1
                return False
            prepared = {**row, "sample_id": f"sft2k-{len(selected):05d}", "source_dataset": c["source"],
                        "source_file": c["filename"], "source_row": c["source_row"],
                        "images": paths, "image_sha256": sorted(hashes)}
            checked = audit_row(prepared, tokenizer, processor, ROOT / "data/source/sft")
            if checked["input_tokens"] > 32768:
                rejected["over_32768_tokens"] += 1
                return False
        except (OSError, ValueError, IndexError, KeyError, SyntaxError) as error:
            rejected["image_or_audit_" + type(error).__name__] += 1
            with (audit_output / "rejected.jsonl").open("a") as stream:
                stream.write(json.dumps({"source": c["source"], "source_row": c["source_row"], "error": str(error)}) + "\n")
            return False
        selected.append(prepared)
        audits[prepared["sample_id"]] = checked
        features_kept[prepared["sample_id"]] = f
        selected_images.update(hashes)
        sources[c["source"]] += 1
        families[f["family"]] += 1
        rounds[f["round_bin"]] += 1
        tool_presence.update(f["tools"])
        if len(selected) % 32 == 0:
            progress = {"selected": len(selected), "families": dict(families), "sources": dict(sources),
                        "round_bins": dict(rounds), "tool_presence": dict(tool_presence), "rejected": dict(rejected)}
            (audit_output / "progress.json").write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
        return True

    def take(predicate, requested):
        queues = {s: deque(c for c in candidates if c["source"] == s and not c["visited"] and predicate(c)) for s in SFT_FILES}
        added = 0
        while added < requested:
            available = [s for s, q in queues.items() if q and sources[s] < 512]
            if not available:
                break
            source = min(available, key=lambda s: (sources[s] / SOURCE_TARGETS[s], s))
            c = queues[source].popleft()
            if c["visited"] or families[c["features"]["family"]] >= FAMILY_TARGETS[c["features"]["family"]]:
                continue
            if accept(c):
                added += 1
        return added

    take(lambda c: "perspective_correct" in c["features"]["tools"], 2048)
    take(lambda c: "sharpen" in c["features"]["tools"], max(0, 48 - tool_presence["sharpen"]))
    take(lambda c: "super_resolution" in c["features"]["tools"], max(0, 128 - tool_presence["super_resolution"]))
    take(lambda c: c["features"]["round_bin"] == "long", max(0, 205 - rounds["long"]))
    for family in ("enhancement", "perception", "retrieval"):
        take(lambda c, family=family: c["features"]["family"] == family and c["features"]["round_bin"] == "medium",
             min(FAMILY_TARGETS[family] - families[family], max(0, 819 - rounds["medium"])))
    for family in ("enhancement", "perception", "retrieval"):
        take(lambda c, family=family: c["features"]["family"] == family and c["features"]["round_bin"] == "short",
             FAMILY_TARGETS[family] - families[family])
        take(lambda c, family=family: c["features"]["family"] == family, FAMILY_TARGETS[family] - families[family])
    if len(selected) != 2048 or dict(families) != FAMILY_TARGETS:
        raise RuntimeError(f"合格轨迹无法满足预定分层，不复制填充：{len(selected)}, {families}")
    rng.shuffle(selected)
    write_jsonl(output / "sft.jsonl", selected)
    write_jsonl(audit_output / "samples.jsonl", [audits[r["sample_id"]] for r in selected])
    lengths = [audits[r["sample_id"]]["input_tokens"] for r in selected]
    keys = ("input_tokens", "visual_tokens", "supervised_tokens", "masked_tokens", "image_count", "assistant_eos")
    summary = {"samples": 2048, "passed_mask_and_shift": 2048, "errors": [], "max_seq_len": 32768,
               "overlength_count": 0, "overlength_ids": [],
               "lengths": {"min": min(lengths), "mean": float(np.mean(lengths)), "median": float(np.median(lengths)),
                           "p90": float(np.quantile(lengths, .9)), "p95": float(np.quantile(lengths, .95)), "max": max(lengths)},
               "totals": {k: sum(audits[r["sample_id"]][k] for r in selected) for k in keys},
               "cpu_audit_passed": True, "remote_lengths_passed": prior["remote_lengths_passed"],
               "remote_length_check": {**prior["remote_length_check"], "method": "reuse earlier checks with identical tokenizer/processor, protocol and template; no new remote requests"},
               "data_sha256": hashlib.sha256((output / "sft.jsonl").read_bytes()).hexdigest(),
               "template_sha256": hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest(),
               "protocol_sha256": hashlib.sha256((ROOT / "protocol.py").read_bytes()).hexdigest(),
               "model_files_sha256": model_hashes,
               "versions": {n: version(n) for n in ("pytrio", "transformers", "numpy", "pillow")},
               "scope": "new 2048 rows independently audited with audit_row; heldout 200/RL128 excluded; no GPU training"}
    if summary["protocol_sha256"] != prior["protocol_sha256"] or summary["template_sha256"] != prior["template_sha256"]:
        raise ValueError("协议或模板改变，不能复用已有远端长度验证")
    manifest = {"seed": 42, "samples": 2048, "data_sha256": summary["data_sha256"], "source_counts": dict(sources),
                "source_revision": "2c1c460af4fa15bd63210cbf426a96664b959944",
                "source_files_sha256": {s: hashlib.sha256((ROOT / "data/source/sft" / s / f).read_bytes()).hexdigest() for s, f in SFT_FILES.items()},
                "source_targets_soft": SOURCE_TARGETS, "source_maximum": 512, "family_counts": dict(families),
                "round_bins": dict(rounds), "tool_trajectory_presence": dict(tool_presence),
                "tool_call_counts": dict(Counter(call["name"] for f in features_kept.values() for call in f["calls"])),
                "rejections": dict(rejected), "frozen_files": frozen, "selection_seconds": time.time() - started,
                "policy": "no duplicate trajectories or image bytes; no repeated-action padding; rare tools prioritized; exact CPU token cap 32768; no character cutoff; original system/tools unchanged",
                "limitations": ["rare tools remain limited by real available examples", "byte hashes miss recompressed copies", "published answer quality not independently re-judged", "source/round targets are soft and actual counts are reported"]}
    for name, expected in frozen.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("选择过程中冻结数据发生变化")
    (audit_output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"complete": True, "manifest": manifest, "audit": summary}), flush=True)


if __name__ == "__main__":
    main()
