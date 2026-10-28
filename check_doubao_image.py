"""用与百炼检查相同的三张图验证豆包 visual、缓存和作者摘要链路。"""

import argparse
import base64
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from PIL import Image
import pytrio as trio

from agent_baseline import ROOT, TokenBudget
from agent_tools import ToolEnvironment, model_image_bytes
from doubao_image import DoubaoImageClient
from service_io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="显式复用已核验回包和缓存，不重发未知状态请求")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError("检查目录已存在，避免重复付费调用")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir.resolve()
    state = json.loads((ROOT / "outputs/base-seven-v1/eval-0001.partial.json").read_text())
    with Image.open(ROOT / "data/source/sft/livevqa/images/interface_v5.png") as image:
        crop = model_image_bytes(image.crop((100, 100, 400, 400)))
    public_url = "https://img.alicdn.com/imgextra/i4/O1CN01YbrnSS1qtmsAkw0Ud_!!6000000005554-2-tps-788-450.png"
    inputs = [("original_base64", base64.b64decode(state["images"]["img_1"])),
              ("public_url", public_url), ("crop_base64", crop)]
    tools = ToolEnvironment(out / "tool-cache", usage_dir=out)
    previous = out / "progress.json"
    old = {r["case"]: r for r in json.loads(previous.read_text())["checks"]} if previous.exists() else {}
    results = []
    for name, image_input in inputs:
        cache = tools.cache_dir / "doubao-image-v1" if name == "public_url" else out / name
        client = DoubaoImageClient(cache)
        cached_before = any(cache.glob("*.result.json"))
        started = time.monotonic()
        result = client.search(image_input, before_submit=lambda: tools.charge("image_search"),
                               on_result=tools.record_doubao_usage)
        elapsed = time.monotonic() - started
        ledger = out / "doubao-usage.jsonl"
        before = ledger.read_bytes()
        assert client.search(image_input, before_submit=lambda: tools.charge("image_search"),
                             on_result=tools.record_doubao_usage) == result
        assert ledger.read_bytes() == before
        seconds = old[name]["seconds"] if name in old else None if cached_before else round(elapsed, 3)
        row = {"case": name, "seconds": seconds, "returned_documents": len(result["matches"]),
               "source_pages": sum(bool(r["source"]) for r in result["matches"]),
               "request_id": result["request_id"], "cache_without_new_request": True,
               "provider_error_code": result.get("provider_error_code", 0),
               "response_kind": result.get("response_kind", "results" if result["matches"] else "empty_search")}
        results.append(row)
        write_json(out / "progress.json", {"checks": results})
        print(json.dumps(row, ensure_ascii=False), flush=True)
    assert len({row["request_id"] for row in results}) == 3

    service = trio.ServiceClient(api_key=os.environ["PYTRIO_API_KEY"])
    sampler = service.create_sampling_client(base_model="Qwen/Qwen3.8-27B")
    tokenizer = sampler.get_tokenizer()
    budget = TokenBudget(out / "summary-tokens.jsonl", 100000)
    def summarize(prompt):
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                              return_dict=False, add_generation_prompt=True, enable_thinking=False)
        response = budget.sample(sampler, trio.ModelInput.from_ints(tokens), "doubao-service-check",
                                 trio.SamplingParams(max_tokens=1024, temperature=.3, top_p=.95, top_k=20, seed=42))
        assert response.input_tokens == len(tokens) and response.sequences[0].text.strip()
        return response.sequences[0].text.strip()
    tools.summarizer, tools.summary_model = summarize, "Qwen/Qwen3.8-27B"
    ledger_before = (out / "doubao-usage.jsonl").read_bytes()
    observation, image = tools.execute({"name": "image_search", "arguments": {"url": public_url}}, {})
    assert image is None and observation.startswith("Tool execution result:")
    assert "No visual matches found." not in observation
    assert (out / "doubao-usage.jsonl").read_bytes() == ledger_before
    summary_before = (out / "summary-tokens.jsonl").read_bytes()
    again, _ = tools.execute({"name": "image_search", "arguments": {"url": public_url}}, {})
    assert again == observation and (out / "summary-tokens.jsonl").read_bytes() == summary_before
    usage = [json.loads(line) for line in ledger_before.splitlines()]
    assert len(usage) == 3
    report = {"provider": "doubao-global-visual", "route": "direct", "ready": True, "checks": results,
              "completed_live_requests": len(usage), "summary_model": tools.summary_model,
              "integrated_summary_characters": len(observation), "summary_cache_verified": True,
              "baseline_question_count": 0,
              "limits": ["three-image connectivity and pipeline check, not recognition accuracy or long-duration SLA",
                         "visual unit price and actual charges not verified against account bill"]}
    write_json(out / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
