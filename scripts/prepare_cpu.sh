#!/usr/bin/env bash
# Run on the NETWORKED CPU instance; the shared path must match the GPU instance.
set -euo pipefail
cd "$(dirname "$0")/.."
# Bootstrap without the CPU image's Python, pip or configured package mirror.
# Keep uv local to the checkout; do not change the user's shell profile.
UV_VERSION="0.13.7"
UV_BIN="$PWD/.bootstrap-linux/bin/uv"
PYPI_INDEX="${TUBE_PYPI_INDEX_URL:-https://pypi.org/simple}"
if [[ "$("$UV_BIN" --version 2>/dev/null || true)" != "uv $UV_VERSION" ]]; then
  mkdir -p .bootstrap-linux
  INSTALLER="$PWD/.bootstrap-linux/uv-installer.sh"
  INSTALLER_URL="https://astral.sh/uv/$UV_VERSION/install.sh"
  echo "Installing standalone uv $UV_VERSION from Astral..."
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error --retry 3 "$INSTALLER_URL" -o "$INSTALLER"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 -O "$INSTALLER" "$INSTALLER_URL"
  else
    echo 'Please install curl or wget on the networked CPU instance.' >&2
    exit 1
  fi
  UV_UNMANAGED_INSTALL="$PWD/.bootstrap-linux/bin" sh "$INSTALLER"
fi
"$UV_BIN" --version
echo "Python package index: $PYPI_INDEX"
# The GPU image may not contain the CPU image's system Python. Keep the actual
# interpreter on the shared filesystem as well as the virtual environment.
export UV_PYTHON_INSTALL_DIR="$PWD/.python-linux"
export UV_CACHE_DIR="$PWD/.uv-cache"
"$UV_BIN" python install 3.13.5 --no-bin
PYTHON_PATH="$("$UV_BIN" python find 3.13.5 --managed-python)"
if [[ ! -x .venv-linux/bin/python ]]; then
  "$UV_BIN" venv .venv-linux --python "$PYTHON_PATH" --seed --default-index "$PYPI_INDEX"
fi
source .venv-linux/bin/activate
python -c 'import sys; assert sys.version_info[:3] == (3,13,5), "Existing .venv-linux is not Python 3.13.5; use a fresh project checkout"'
python -m pip install --upgrade pip --index-url "$PYPI_INDEX"
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -r requirements-model.txt 'pytest>=8' --index-url "$PYPI_INDEX"
python -m pip check
python -m pip freeze > .venv-linux/installed-packages.txt
python -m tube_agent.doctor --model "${TUBE_MODEL_PATH:-/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/assets/model}"
python -m pytest -q
echo 'CPU preparation complete. On the GPU instance: source .venv-linux/bin/activate'
