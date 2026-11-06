#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${TUBE_PYTHON_BIN:-/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/.runtime/py311/bin/python}"
if [[ ! -x "$PY" ]]; then echo "Set TUBE_PYTHON_BIN to the shared Python 3.11 executable." >&2; exit 1; fi
"$PY" -c 'import sys,platform; assert sys.version_info[:2]==(3,11) and sys.platform=="linux" and platform.machine()=="x86_64", "Use shared Linux x86_64 Python 3.11"'
"$PY" scripts/offline_bundle.py verify
"$PY" - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path("MANIFEST.json").read_text())
expected = {name for name in manifest["files"] if name.startswith("sam-wheels/") and name.endswith(".whl")}
actual = {p.as_posix() for p in Path("sam-wheels").glob("*.whl")}
if not expected or expected != actual:
    raise SystemExit("SAM wheel inventory mismatch. Re-extract the verified lab patch before installing.")
print("Verified SAM wheel inventory:", len(expected))
PY
if [[ ! -x .venv-sam/bin/python ]]; then "$PY" -m venv .venv-sam; fi
.venv-sam/bin/python scripts/reuse_torch.py --source-python "$PY"
export PIP_CONFIG_FILE=/dev/null PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
.venv-sam/bin/python -m pip install --no-index --only-binary=:all: --find-links sam-wheels --find-links wheels \
  --constraint .venv-sam/torch-constraints.txt --requirement requirements-sam.txt
.venv-sam/bin/python -m pip check
.venv-sam/bin/python -c 'import torch, numpy, timm, scipy, pycocotools, einops; from sam3.model_builder import build_sam3_predictor; print("SAM imports OK", torch.__version__, numpy.__version__)'
echo "Prepared isolated .venv-sam. Existing .venv-linux and DiG were not changed."
