#!/usr/bin/env bash
# Run on the NETWORKED CPU instance; the shared path must match the GPU instance.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
# The GPU image may not contain the CPU image's system Python. Keep the actual
# interpreter on the shared filesystem as well as the virtual environment.
"$PYTHON_BIN" -m venv .bootstrap-linux
.bootstrap-linux/bin/python -m pip install uv==0.13.7
export UV_PYTHON_INSTALL_DIR="$PWD/.python-linux"
export UV_CACHE_DIR="$PWD/.uv-cache"
UV_BIN="$PWD/.bootstrap-linux/bin/uv"
"$UV_BIN" python install 3.13.5 --no-bin
PYTHON_PATH="$("$UV_BIN" python find 3.13.5 --managed-python)"
if [[ ! -x .venv-linux/bin/python ]]; then
  "$UV_BIN" venv .venv-linux --python "$PYTHON_PATH" --seed
fi
source .venv-linux/bin/activate
python -c 'import sys; assert sys.version_info[:3] == (3,13,5), "Existing .venv-linux is not Python 3.13.5; use a fresh project checkout"'
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -r requirements-model.txt 'pytest>=8'
python -m pip check
python -m pip freeze > .venv-linux/installed-packages.txt
python -m tube_agent.doctor --model "${TUBE_MODEL_PATH:-/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/assets/model}"
python -m pytest -q
echo 'CPU preparation complete. On the GPU instance: source .venv-linux/bin/activate'
