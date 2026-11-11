#!/usr/bin/env bash
# Usage from the existing training checkout: bash .../with_mapped_egl.sh AUDIT_DIR COMMAND...
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: with_mapped_egl.sh AUDIT_DIR COMMAND [ARGS...]" >&2
  exit 2
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project="$(cd "${ZAP_PROJECT_ROOT:-$PWD}" && pwd -P)"
mkdir -p "$1"
export ZAP_EGL_AUDIT_DIR="$(cd "$1" && pwd -P)"
shift
test -f "$project/.runtime/egl-jammy/EGL_RUNTIME.json"
"$project/.venv-linux/bin/python" "$project/scripts/check_render.py" --verify-runtime
export LD_LIBRARY_PATH="$project/.runtime/egl-jammy/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl TUBE_MUJOCO_GL=egl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$here/startup:$here:$project${PYTHONPATH:+:$PYTHONPATH}"
"$project/.venv-linux/bin/python" -c 'import os, sitecustomize, sys; sys.exit(0 if os.path.samefile(sitecustomize.__file__, sys.argv[1]) else "EGL startup hook is shadowed")' "$here/startup/sitecustomize.py"
exec "$@"
