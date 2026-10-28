"""在 CPU 服务器直连验证国内服务与真实七工具，不运行 benchmark。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from PIL import Image
import pytrio as trio
from transformers import AutoImageProcessor

from agent_baseline import ROOT, TokenBudget, sample_prompt
from agent_tools import ToolEnvironment, model_image_bytes
from doubao_image import DoubaoImageClient
from deepseek_search import DeepSeekSearchClient, DeepSeekPageUnavailable
from judge_baseline import JUDGE_PROMPT, judge_record
from paddle_ocr import OCRServiceError, PaddleOCRClient
from protocol import IMAGE_MARKER
from service_io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(name, None)
    os.environ["PADDLE_OCR_PROXY"] = ""
    for name in ("PYTRIO_API_KEY", "DEEPSEEK_API_KEY", "DOUBAO_SEARCH_API_KEY", "PADDLE_OCR_TOKEN", "JUDGE_API_KEY"):
        if not os.environ.get(name, "").strip():
            raise ValueError(f"缺少服务配置: {name}")
    if args.output_dir.exists():
        raise FileExistsError("验证目录已存在，请先核对记录，避免无意重复付费测试")
    args.output_dir.mkdir(parents=True)
    out = args.output_dir.resolve()
    results = []
    started = time.time()

    def check(name, function):
        before = time.monotonic()
        try:
            detail = function()
            row = {"name": name, "ok": True, "seconds": round(time.monotonic() - before, 3), "detail": detail}
        except Exception as error:
            # 诊断器收集各独立服务的结果；异常仍导致最终退出码非零。
            message = str(error)
            for key, value in os.environ.items():
                if value and len(value) >= 8 and (key.endswith("_KEY") or key.endswith("_TOKEN")):
                    message = message.replace(value, "[REDACTED]")
            row = {"name": name, "ok": False, "seconds": round(time.monotonic() - before, 3),
                   "error_type": type(error).__name__, "error": message[:800]}
        results.append(row)
        write_json(out / "progress.json", {"checks": results})
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    source = ROOT / "data/source/sft/livevqa/images/interface_v5.png"
    with Image.open(source) as image:
        screenshot = model_image_bytes(image)
        small = model_image_bytes(image.crop((0, 0, 64, 64)))
        crop = model_image_bytes(image.crop((100, 100, 400, 400)))
    tools = ToolEnvironment(out / "tool-cache", ocr_cache_dir=out / "ocr-cache", usage_dir=out)
    images = {"img_1": small}
    for name, arguments in (("crop", {"image": "img_1", "x": 0, "y": 0, "width": 48, "height": 48}),
                            ("sharpen", {"image": "img_1"}), ("perspective_correct", {"image": "img_1"}),
                            ("super_resolution", {"image": "img_1", "scale": 4})):
        def image_check(name=name, arguments=arguments):
            message, raw = tools.execute({"name": name, "arguments": arguments}, images)
            with Image.open(io.BytesIO(raw)) as result:
                size = result.size
            if name == "super_resolution":
                assert size == (256, 256)
            return {"size": size, "message": message}
        check(name, image_check)

    web = DeepSeekSearchClient(out / "deepseek-http", out)
    queries = ["Kessock Bridge opening year", "故宫博物院 建院 时间", "Eiffel Tower opening date"]
    urls = ["https://docs.python.org/3/library/json.html", "https://www.python.org/about/", "https://www.gov.uk/"]
    for index in range(args.rounds):
        def search_check(index=index):
            rows = web.search(queries[index % len(queries)], 3)
            assert rows and all(row["title"] and row["link"] for row in rows)
            return {"results": len(rows), "with_full_text": sum(bool(row["content"].strip()) for row in rows)}
        check(f"deepseek_search_{index}", search_check)
        def fetch_check(index=index):
            row = web.fetch(urls[index % len(urls)])
            assert len(row["content"].strip()) > 20
            return {"url": urls[index % len(urls)], "content_characters": len(row["content"])}
        check(f"deepseek_fetch_{index}", fetch_check)

    def missing_page_check():
        try:
            web.fetch("https://example.invalid/does-not-exist")
        except DeepSeekPageUnavailable:
            return {"unavailable_page_is_explicit": True}
        raise AssertionError("无效页面未返回明确不可用状态")
    check("fetch_unavailable_page", missing_page_check)

    for index in range(args.rounds):
        def ocr_check(index=index):
            with Image.open(source) as image:
                raw = model_image_bytes(image.crop((index, 0, image.width, image.height)))
            cache = out / f"ocr-round-{index}"
            options = {"useDocOrientationClassify": False, "useDocUnwarping": False, "useChartRecognition": False}
            try:
                PaddleOCRClient(cache, timeout=.001).parse(raw, options)
            except OCRServiceError as error:
                assert "等待超时" in str(error), str(error)
            assert len(list(cache.glob("*.job.json"))) == 1
            ledger_before = (cache / "submissions.jsonl").read_bytes()
            result = PaddleOCRClient(cache).parse(raw, options)
            assert (cache / "submissions.jsonl").read_bytes() == ledger_before
            assert len(ledger_before.splitlines()) == 1 and result["formatted_text"].strip()
            before = (cache / "submissions.jsonl").read_bytes()
            again = PaddleOCRClient(cache).parse(raw, options)
            assert again == result and (cache / "submissions.jsonl").read_bytes() == before
            return {"characters": len(result["formatted_text"]), "resume_without_resubmission": True,
                    "cache_without_resubmission": True, "job_submissions": 1}
        check(f"paddle_ocr_{index}", ocr_check)

    original = json.loads((ROOT / "outputs/base-seven-v1/eval-0001.partial.json").read_text())
    image_inputs = [base64.b64decode(original["images"]["img_1"]),
                    "https://img.alicdn.com/imgextra/i4/O1CN01YbrnSS1qtmsAkw0Ud_!!6000000005554-2-tps-788-450.png", crop]
    image_result = None
    for index in range(args.rounds):
        def image_search_check(index=index):
            nonlocal image_result
            cache = out / f"doubao-round-{index}"
            image_client = DoubaoImageClient(cache)
            image_result = image_client.search(image_inputs[index % len(image_inputs)], before_submit=lambda: None,
                                               on_result=tools.record_doubao_usage)
            assert image_result["request_id"]
            ledger = out / "doubao-usage.jsonl"
            before = ledger.read_bytes()
            cached = image_client.search(image_inputs[index % len(image_inputs)], before_submit=lambda: None,
                                         on_result=tools.record_doubao_usage)
            assert cached == image_result and ledger.read_bytes() == before
            return {"matches": len(image_result["matches"]), "request_id": image_result["request_id"],
                    "source_pages": sum(bool(r["source"]) for r in image_result["matches"]),
                    "cache_without_new_usage": True}
        check(f"doubao_image_search_{index}", image_search_check)

    service = None
    def create_clients():
        nonlocal service
        service = trio.ServiceClient(api_key=os.environ["PYTRIO_API_KEY"])
        return {"authenticated_directly": True}
    authenticated = check("pytrio_authentication", create_clients)
    if authenticated["ok"]:
        summary_sampler = service.create_sampling_client(base_model="Qwen/Qwen3.8-27B")
        summary_tokenizer = summary_sampler.get_tokenizer()
        budget = TokenBudget(out / "summary-tokens.jsonl", 150000)
        def summarize(prompt):
            tokens = summary_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                            return_dict=False, add_generation_prompt=True, enable_thinking=False)
            result = budget.sample(summary_sampler, trio.ModelInput.from_ints(tokens), "service-check",
                                   trio.SamplingParams(max_tokens=1024, temperature=.3, top_p=.95, top_k=20, seed=42))
            assert result.input_tokens == len(tokens) and result.sequences[0].text.strip()
            return result.sequences[0].text.strip()
        tools.summarizer, tools.summary_model = summarize, "Qwen/Qwen3.8-27B"
        def integrated_text_check():
            action = {"name": "text_search", "arguments": {"q": queries[0], "hl": "en", "top_k": 1}}
            message, _ = tools.execute(action, images)
            ledger = out / "deepseek-usage.jsonl"
            before = ledger.read_bytes()
            action["name"] = "web_search"
            alias, _ = tools.execute(action, images)
            assert message == alias and "Summary:" in message and ledger.read_bytes() == before
            return {"summary_characters": len(message), "alias_matches": True, "cache_verified": True}
        check("integrated_text_and_alias", integrated_text_check)
        def integrated_image_check():
            value, _ = tools.execute({"name": "image_search", "arguments": {"url": image_inputs[1]}}, images)
            assert value.startswith("Tool execution result:") and "No visual matches found." not in value and len(value) > 80
            return {"summary_characters": len(value)}
        check("integrated_image_summary", integrated_image_check)
        main_sampler = service.create_sampling_client(base_model="Qwen/Qwen3.5-4B")
        main_tokenizer = main_sampler.get_tokenizer()
        model_dir = Path.home() / ".cache/modelscope/models/Qwen--Qwen3.5-4B/snapshots/master"
        processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False, local_files_only=True)
        main_budget = TokenBudget(out / "main-tokens.jsonl", 100000)
        for index in range(args.rounds):
            def main_check(index=index):
                turns = [["system", "You are a helpful assistant.", None],
                         ["user", IMAGE_MARKER + "This is a service connection check. Reply briefly with READY.", base64.b64encode(small).decode()]]
                prompt = sample_prompt(main_tokenizer, turns, processor)
                response = main_budget.sample(main_sampler, prompt, f"service-main-{index}",
                                              trio.SamplingParams(max_tokens=256, temperature=0, seed=42))
                assert response.input_tokens == len(prompt) and response.sequences[0].text.strip()
                return {"input_tokens": response.input_tokens, "output_tokens": len(response.sequences[0].tokens),
                        "image_token_alignment": True}
            check(f"pytrio_multimodal_{index}", main_check)

    judge_dir = out / "judge"
    judge_dir.mkdir()
    config = {"model": os.environ["JUDGE_MODEL"], "base_url": os.environ["JUDGE_BASE_URL"],
              "protocol": "anthropic", "temperature": 0, "max_tokens": 1024,
              "prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()}
    config["sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    for index in range(args.rounds):
        def judge_check(index=index):
            record = {"sample_id": f"service-judge-{index}", "benchmark": "service-check",
                      "question": f"What is {index + 1} + 1?", "reference": str(index + 2), "answer": str(index + 2),
                      "termination": "final_answer"}
            result = judge_record(record, judge_dir, config)
            assert result["correct"]
            before = (judge_dir / "requests.jsonl").read_bytes()
            assert judge_record(record, judge_dir, config) == result
            assert (judge_dir / "requests.jsonl").read_bytes() == before
            return {"correct": result["correct"], "usage": result["usage"], "cache_verified": True}
        check(f"glm_judge_{index}", judge_check)

    failed = [row["name"] for row in results if not row["ok"]]
    report = {"started_at": started, "finished_at": time.time(), "rounds": args.rounds, "checks": results,
              "passed": len(results) - len(failed), "failed": failed, "ready": not failed,
              "route": "direct; proxy variables removed; OCR proxy disabled", "baseline_question_count": 0,
              "limitations": ["small live batch, not a long-duration availability guarantee",
                              "site-specific fetch failures remain explicit; client usage is not an actual bill"]}
    write_json(out / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
