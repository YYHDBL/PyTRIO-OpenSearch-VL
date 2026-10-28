#!/usr/bin/env bash
# 七工具协议新 baseline：独立输出目录，不写入已停止的旧三工具目录。
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs
# 此版本全部从服务器直连，不继承旧的 Mac 反向代理。
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export PADDLE_OCR_PROXY=
trap 'printf "%s\n" "$?" > logs/base-aligned-100-v1.exit' EXIT

.venv/bin/python -u agent_baseline.py \
  --data data/selected-eval100-v1/eval.jsonl --output-dir outputs/base-aligned-100-v1 \
  --max-context 65536 --max-tokens 32768 --max-turns 50 \
  --token-budget 25000000 --summary-token-budget 8000000 --image-request-budget 800 \
  > logs/base-aligned-100-v1.log 2>&1
.venv/bin/python -u judge_baseline.py \
  --results-dir outputs/base-aligned-100-v1 --output-dir outputs/judge-base-aligned-100-v1 \
  --expected 100 > logs/judge-base-aligned-100-v1.log 2>&1
