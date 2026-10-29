#!/usr/bin/env bash
# 固定最终 SFT Sampler，与 Base 共用数据、工具和生成设置。
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export PADDLE_OCR_PROXY=
trap 'printf "%s\n" "$?" > logs/sft2048-aligned-100-v2.exit' EXIT

sft_sampler_path=$(.venv/bin/python -c 'import json; from pathlib import Path; r=json.loads(Path("outputs/sft-2048-v1/sft_result.json").read_text()); assert r["complete"] and r["steps"] == 512 and r["samples"] == 2048; assert "/sampler_weights/" in r["sampler_path"]; print(r["sampler_path"])')

.venv/bin/python -u agent_baseline.py \
  --data data/selected-eval100-v1/eval.jsonl --output-dir outputs/sft2048-aligned-100-v2 \
  --model Qwen/Qwen3.5-4B --model-path "$sft_sampler_path" \
  --max-context 65536 --max-tokens 32768 --max-turns 50 \
  --token-budget 25000000 --summary-token-budget 8000000 --image-request-budget 800 \
  > logs/sft2048-aligned-100-v2.log 2>&1
.venv/bin/python -u judge_baseline.py \
  --results-dir outputs/sft2048-aligned-100-v2 --output-dir outputs/judge-sft2048-aligned-100-v2 \
  --expected 100 > logs/judge-sft2048-aligned-100-v2.log 2>&1
