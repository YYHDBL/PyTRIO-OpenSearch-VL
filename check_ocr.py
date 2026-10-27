"""用服务器上的一张真实 SFT 图片验证 OCR 调用与缓存，不运行语言模型。"""

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv

from agent_tools import ToolEnvironment
from paddle_ocr import PaddleOCRClient


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, default=root / "outputs/ocr-smoke-v1.json")
    parser.add_argument("--require-text", action="store_true")
    args = parser.parse_args()
    load_dotenv(root / ".env")
    if args.image:
        image_path, sample_id, source = args.image, None, "provided_image"
    else:
        rows = (json.loads(line) for line in (root / "data/selected/sft.jsonl").open())
        row = next(row for row in rows if row["source_dataset"] == "webqa")
        image_path, sample_id, source = Path(row["images"][0]), row["sample_id"], row["source_dataset"]
    raw = image_path.read_bytes()
    tools = ToolEnvironment(root / "data/tool_cache")
    action = {"name": "layout_parsing", "arguments": {"image": "img_1"}}
    start = time.monotonic()
    report, image = tools.execute(action, {"img_1": raw})
    elapsed = time.monotonic() - start
    if image is not None or "ALL RECOGNIZED TEXT:" not in report:
        raise ValueError("OCR 工具输出与预期结构不一致")
    cache = root / "data/tool_cache/ocr-v1"
    ledger = cache / "submissions.jsonl"
    before = ledger.read_bytes() if ledger.exists() else b""
    result = PaddleOCRClient(cache).parse(raw, {"useDocOrientationClassify": False,
                                              "useDocUnwarping": False, "useChartRecognition": False})
    after = ledger.read_bytes() if ledger.exists() else b""
    if before != after:
        raise ValueError("重复 OCR 请求未使用缓存")
    if args.require_text and not result["formatted_text"].strip():
        raise ValueError("测试图片未提取出文字")
    summary = {"sample_id": sample_id, "source": source, "model": result["model"],
               "image_sha256": hashlib.sha256(raw).hexdigest(), "page_count": result["page_count"],
               "block_labels": dict(Counter(block["block_label"] for block in result["blocks"])),
               "recognized_text_blocks": len(result["rec_texts"]), "text_characters": len(result["formatted_text"]),
               "attempt_seconds": round(elapsed, 2), "cache_reused_without_new_submission": True,
               "language_model_requests": 0,
               "limitation": "PaddleOCR-VL-1.6 replaces the author's unspecified layout backend"}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
