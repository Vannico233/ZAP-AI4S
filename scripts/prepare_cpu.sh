#!/usr/bin/env bash
# Run on the NETWORKED CPU instance; the shared path must match the GPU instance.
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse a Python installation on the CPU/GPU shared mount. No uv or Python
# download is needed. A venv still depends on its base interpreter's files.
PYTHON_BIN="${TUBE_PYTHON_BIN:-python}"
SHARED_ROOT="${TUBE_SHARED_ROOT:-$(dirname "$PWD")}"
PYPI_INDEX="${TUBE_PYPI_INDEX_URL:-https://pypi.org/simple}"
TORCH_INDEX="${TUBE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

check_python() {
  "$1" - "$SHARED_ROOT" <<'PY'
import sys
from pathlib import Path

if not (3, 10) <= sys.version_info[:2] <= (3, 13):
    raise SystemExit("Use Python 3.10-3.13; recommended: the existing shared Python 3.11.")
shared = Path(sys.argv[1]).resolve()
base = Path(sys.base_prefix).resolve()
project = Path.cwd().resolve()
if not base.is_relative_to(shared) or not project.is_relative_to(shared):
    raise SystemExit(
        f"Python base: {base}\nProject: {project}\nShared root: {shared}\n"
        "Both must be on the CPU/GPU shared mount. Set TUBE_PYTHON_BIN to a shared "
        "Python executable, or TUBE_SHARED_ROOT to the actual shared parent directory."
    )
print(f"Python: {sys.version.split()[0]}\nExecutable: {sys.executable}\nBase: {base}")
PY
}

if [[ -e .venv-linux || -L .venv-linux ]]; then
  if [[ ! -x .venv-linux/bin/python ]]; then
    echo '.venv-linux exists but its Python is missing. Move this environment aside before retrying.' >&2
    exit 1
  fi
  check_python "$PWD/.venv-linux/bin/python"
else
  check_python "$PYTHON_BIN"
  "$PYTHON_BIN" -m venv .venv-linux
fi
source .venv-linux/bin/activate
# Recover if a previous venv creation stopped before installing pip.
if ! python -m pip --version >/dev/null 2>&1; then
  python -m ensurepip --upgrade
fi
echo "Python package index: $PYPI_INDEX"
echo "PyTorch CUDA 12.8 index: $TORCH_INDEX"
python -m pip install --upgrade pip --index-url "$PYPI_INDEX" --retries 5 --timeout 60
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url "$TORCH_INDEX" --retries 5 --timeout 60
python -m pip install -r requirements.txt -r requirements-model.txt 'pytest>=8' --index-url "$PYPI_INDEX" --retries 5 --timeout 60
python -m pip check
python -m pip freeze > .venv-linux/installed-packages.txt
python -m tube_agent.doctor --model "${TUBE_MODEL_PATH:-/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/assets/model}"
python -m pytest -q
echo 'CPU preparation complete. On the GPU instance: source .venv-linux/bin/activate'
