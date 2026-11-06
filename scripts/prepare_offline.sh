#!/usr/bin/env bash
# Run inside the uploaded, extracted bundle. No git or package downloads needed.
set -euo pipefail
cd "$(dirname "$0")/.."
export TUBE_PYTHON_BIN="${TUBE_PYTHON_BIN:-/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/.runtime/py311/bin/python}"
"$TUBE_PYTHON_BIN" scripts/offline_bundle.py verify --check-target
export TUBE_OFFLINE_BUNDLE=1
export TUBE_TORCH_MODE=reuse
export PIP_NO_INDEX=1
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec bash scripts/prepare_cpu.sh
