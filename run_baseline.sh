#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs
trap 'printf "%s\n" "$?" > logs/baseline.exit' EXIT

.venv/bin/python -u agent_baseline.py --output-dir outputs/base-v1
.venv/bin/python -u judge_baseline.py \
  --results-dir outputs/base-v1 --output-dir outputs/judge-base-v1
