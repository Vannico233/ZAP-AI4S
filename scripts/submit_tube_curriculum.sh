#!/usr/bin/env sh
# Paste this script into the GPU job entrypoint, or run it with sh.
exec /usr/bin/env bash -c '
set -eo pipefail

PROJECT_ROOT="/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/ZAP-AI4S-git"
MODEL_DIR="/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/assets/model"
cd "$PROJECT_ROOT"

{
    source "$PROJECT_ROOT/.venv-linux/bin/activate"
    export PATH="$PROJECT_ROOT/.venv-linux/bin:$PATH"
    export PYTHONUNBUFFERED=1

    command -v python
    python -c "import sys; print(\"Python:\", sys.version); print(\"Environment:\", sys.prefix)"

    bash "$PROJECT_ROOT/scripts/start_tube_curriculum.sh" \
        runs/tube_curriculum/v1 \
        --model "$MODEL_DIR"
} 2>&1 | tee -a "$PROJECT_ROOT/start_self.log"
'
