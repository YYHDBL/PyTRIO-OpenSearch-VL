"""在 CPU 服务器用真实图片检查七类工具及训练/评测提示词对齐。"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from PIL import Image
import pytrio as trio
from transformers import AutoImageProcessor, AutoTokenizer

from agent_baseline import ROOT, TokenBudget, author_system, sample_prompt
from agent_tools import ToolEnvironment, model_image_bytes
from protocol import IMAGE_MARKER, IMAGE_PAD, TEMPLATE_PATH, build_messages


def check_prompt(rows: list[dict], tokenizer, processor) -> dict:
    row = next(row for row in rows if len(row["images"]) > 1)
    messages = build_messages(row, ROOT / "data/source/sft")
    system = author_system(rows)
    if messages[0]["content"] != system:
        raise ValueError("训练与评测 system 和 tools 不一致")
    turns = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            turns.append([message["role"], content, None])
            continue
        images = [item["image"] for item in content if item["type"] == "image"]
        if len(images) != 1:
            raise ValueError("单条消息的图片数量与当前轨迹协议不一致")
        text = "".join(IMAGE_MARKER if item["type"] == "image" else item["text"] for item in content)
        with Image.open(images[0]) as image:
            raw = model_image_bytes(image)
        turns.append([message["role"], text, base64.b64encode(raw).decode("ascii")])
    prompt = sample_prompt(tokenizer, turns, processor)
    actual = []
    for chunk in prompt.chunks:
        if isinstance(chunk, trio.ImageChunk):
            actual.append(tokenizer.convert_tokens_to_ids(IMAGE_PAD))
        else:
            actual.extend(chunk.tokens)
    expected = tokenizer.apply_chat_template(messages, chat_template=TEMPLATE_PATH.read_text(), tokenize=True,
                                             return_dict=False, add_generation_prompt=True, enable_thinking=True)
    if actual != list(expected):
        raise ValueError("训练与评测的图文消息顺序或 token 编码不一致")
    return {"sample_id": row["sample_id"], "images": len(row["images"]),
            "messages": len(messages), "template_tokens_match": True,
            "system_and_tools_match": True, "input_tokens": len(prompt)}


def main() -> None:
    load_dotenv(ROOT / ".env")
    rows = [json.loads(line) for line in (ROOT / "data/selected/sft.jsonl").open()]
    model_dir = Path.home() / ".cache/modelscope/models/Qwen--Qwen3.5-4B/snapshots/master"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False, local_files_only=True)
    prompt_check = check_prompt(rows, tokenizer, processor)
    image_path = ROOT / "data/source/sft/livevqa/images/interface_v5.png"
    with Image.open(image_path) as source:
        image_bytes = model_image_bytes(source.crop((0, 0, 64, 64)))
    images = {"img_1": image_bytes}
    tools = ToolEnvironment(ROOT / "data/tool_cache/seven-tool-check")
    checks = {}
    started = time.monotonic()
    for name, arguments in (
        ("crop", {"image": "img_1", "x": 0, "y": 0, "width": 48, "height": 48}),
        ("sharpen", {"image": "img_1", "amount": 1.5}),
        ("perspective_correct", {"image": "img_1"}),
        ("super_resolution", {"image": "img_1", "scale": 4}),
    ):
        message, output = tools.execute({"name": name, "arguments": arguments}, images)
        with Image.open(io.BytesIO(output)) as image:
            width, height = image.size
        if name == "super_resolution" and (width, height) != (256, 256):
            raise ValueError("EDSR 没有输出真实的 4 倍图像")
        checks[name] = {"width": width, "height": height, "message": message}
        print(name, "ok", width, height, flush=True)
    tools.ocr_cache_dir = ROOT / "data/tool_cache/ocr-v1"
    ocr, _ = tools.execute({"name": "layout_parsing", "arguments": {"image": "img_1"}},
                           {"img_1": image_path.read_bytes()})
    if "ALL RECOGNIZED TEXT:" not in ocr or len(ocr.split("ALL RECOGNIZED TEXT:", 1)[1].strip()) < 1:
        raise ValueError("OCR 没有返回真实文本")
    checks["layout_parsing"] = {"recognized_characters": len(ocr.split("ALL RECOGNIZED TEXT:", 1)[1].strip())}
    print("layout_parsing ok", flush=True)
    # 实测工具摘要调用单独记账；每次运行使用独立小额度，避免影响正式评测缓存。
    service = trio.ServiceClient(api_key=os.environ["PYTRIO_API_KEY"])
    sampler = service.create_sampling_client(base_model="Qwen/Qwen3.8-27B")
    summary_tokenizer = sampler.get_tokenizer()
    check_dir = ROOT / "outputs/seven-tool-check"
    check_dir.mkdir(parents=True, exist_ok=True)
    budget = TokenBudget(check_dir / "summary-tokens.jsonl", 150000)
    def summarize(prompt: str) -> str:
        ids = summary_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                    return_dict=False, add_generation_prompt=True, enable_thinking=False)
        result = budget.sample(sampler, trio.ModelInput.from_ints(ids), "tool-check",
                               trio.SamplingParams(max_tokens=1024, temperature=0.3, top_p=.95, top_k=20, seed=42))
        value = result.sequences[0].text.strip()
        if not value:
            raise ValueError("摘要模型返回空内容")
        return value
    tools.summarizer = summarize
    tools.summary_model = "Qwen/Qwen3.8-27B"
    search_args = {"q": "Kessock Bridge opening year", "hl": "en", "top_k": 1}
    text, _ = tools.execute({"name": "text_search", "arguments": search_args}, images)
    alias, _ = tools.execute({"name": "web_search", "arguments": search_args}, images)
    if text != alias or "Summary:" not in text:
        raise ValueError("文字搜索与 web_search 别名结果不一致")
    checks["text_search_and_web_search"] = {"summary_present": True, "alias_matches": True}
    print("text_search and web_search ok", flush=True)
    lens, _ = tools.execute({"name": "image_search", "arguments": {"url": "img_1"}},
                            {"img_1": image_path.read_bytes()})
    if not lens.startswith("Tool execution result:") or "No visual matches found." in lens:
        raise ValueError("图片搜索没有返回摘要")
    checks["image_search"] = {"summary_present": True}
    print("image_search ok", flush=True)
    summary = {"prompt": prompt_check, "tools": checks, "summary_model": "Qwen/Qwen3.8-27B",
               "ocr_model": "PaddleOCR-VL-1.6", "elapsed_seconds": round(time.monotonic() - started, 2),
               "tool_stats": tools.stats, "baseline_question_count": 0}
    (check_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
