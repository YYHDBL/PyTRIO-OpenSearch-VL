#!/usr/bin/env bash
# 固定SFT模型的训练题难度诊断，不创建训练客户端。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export PADDLE_OCR_PROXY=
trap 'printf "%s\n" "$?" > logs/rl-diagnostic-128x4-v1.exit' EXIT
.venv/bin/python -u rl_diagnostic.py --output-dir outputs/rl-diagnostic-128x4-v1
