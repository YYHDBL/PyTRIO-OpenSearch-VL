#!/usr/bin/env bash
# 独立的2题链路实测；只允许一个batch和一次参数更新。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export PADDLE_OCR_PROXY=
trap 'printf "%s\n" "$?" > logs/rl-smoke-2x4-v2.exit' EXIT
.venv/bin/python -u train_rl.py --execute --smoke \
  --output-dir outputs/rl-smoke-2x4-v2/run \
  --selected-ids outputs/rl-smoke-2x4-v2/ids.json \
  --aggregation seq-mean-token-sum --token-budget 1000000
