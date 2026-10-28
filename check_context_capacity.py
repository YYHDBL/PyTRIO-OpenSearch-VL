"""低输出量探测远程采样窗口，独立于 benchmark 和模型效果评估。"""

import argparse
import io
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from PIL import Image
import pytrio as trio
from transformers import AutoImageProcessor

from service_io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    if args.output_dir.exists():
        raise FileExistsError("探测目录已存在，不重复提交")
    args.output_dir.mkdir(parents=True)
    client = trio.ServiceClient(api_key=os.environ["PYTRIO_API_KEY"])
    capability = next(m for m in client.get_server_capabilities().supported_models if m.model_name == "Qwen/Qwen3.5-4B")
    write_json(args.output_dir / "capabilities.json", capability.model_dump())
    sampler = client.create_sampling_client(base_model="Qwen/Qwen3.5-4B")
    tokenizer = sampler.get_tokenizer()
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    pad = encode(" test")[0]
    suffix = encode("\nIgnore the padding. Reply only OK.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    records = []
    for name, target, max_new, vision in (("declared_32k_output", 256, 32768, False),
                                          ("input_over_32k", 33000, 8, False),
                                          ("vision_32k_plus_32k", 32000, 32768, True),
                                          ("input_65k", 65000, 8, False)):
        prefix = encode("<|im_start|>user\n")
        middle = []
        if vision:
            with Image.open(root / "data/source/sft/livevqa/images/interface_v5.png") as source:
                image = source.convert("RGB").crop((0, 0, 64, 64))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            model_dir = Path.home() / ".cache/modelscope/models/Qwen--Qwen3.5-4B/snapshots/master"
            processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
            count = processor.get_number_of_image_patches(image.height, image.width, images_kwargs={}) // int(processor.merge_size) ** 2
            prefix += encode("<|vision_start|>")
            middle = [trio.ImageChunk(data=stream.getvalue(), format="png", expected_tokens=count),
                      trio.types.EncodedTextChunk(tokens=encode("<|vision_end|>\n"))]
        chunks = [trio.types.EncodedTextChunk(tokens=prefix), *middle]
        base = len(trio.ModelInput(chunks=chunks)) + len(suffix)
        prompt = trio.ModelInput(chunks=[*chunks, trio.types.EncodedTextChunk(tokens=[pad] * (target - base) + suffix)])
        assert len(prompt) == target and target + max_new <= capability.sampling.max_seq_len
        marker = args.output_dir / (name + ".json")
        row = {"case": name, "input_tokens_requested": target, "max_new_tokens_requested": max_new,
               "vision": vision, "at": time.time(), "status": "submitted", "purpose": "API capacity, not benchmark"}
        write_json(marker, row)
        start = time.monotonic()
        try:
            result = sampler.sample(prompt=prompt, num_samples=1,
                                    sampling_params=trio.SamplingParams(max_tokens=max_new, temperature=0, seed=42)).result()
            assert result.input_tokens == target
            row.update(status="accepted", input_tokens=result.input_tokens, output_tokens=len(result.sequences[0].tokens),
                       seconds=round(time.monotonic() - start, 3), sequence_fields=list(vars(result.sequences[0])))
        except Exception as error:
            message = str(error)
            for key, value in os.environ.items():
                if value and len(value) >= 8 and (key.endswith("_KEY") or key.endswith("_TOKEN")):
                    message = message.replace(value, "[REDACTED]")
            row.update(status="failed", error_type=type(error).__name__, error=message[:1200],
                       note="submitted requests may incur usage; no automatic retry")
        write_json(marker, row)
        records.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    write_json(args.output_dir / "summary.json", {"model": capability.model_name,
               "advertised_model_context": capability.max_context_length,
               "advertised_sampling_seq_len": capability.sampling.max_seq_len, "checks": records,
               "ready": all(r["status"] == "accepted" for r in records), "benchmark_questions": 0})


if __name__ == "__main__":
    main()
