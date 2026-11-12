#!/usr/bin/env sh
# Run with sh, or paste into a job whose cwd or ZAP_PROJECT_ROOT is the checkout.
set -eu
if [ -n "${ZAP_PROJECT_ROOT:-}" ]; then
    checkout=$ZAP_PROJECT_ROOT
elif [ "${0##*/}" = submit_tube_curriculum.sh ] && [ -f "$0" ]; then
    checkout=$(dirname "$0")/..
else
    checkout=$PWD
fi
cd "$checkout"
if [ ! -f configs/tube_curriculum_bimanual.json ] || [ ! -f scripts/start_tube_curriculum.sh ]; then
    echo "Expected the bimanual checkout at $PWD. Set ZAP_PROJECT_ROOT to its directory." >&2
    exit 1
fi
export ZAP_PROJECT_ROOT="$(pwd -P)"
exec /usr/bin/env bash -c '
set -eo pipefail

PROJECT_ROOT="$ZAP_PROJECT_ROOT"
MODEL_DIR="/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/assets/model"
RUN_DIR="${TUBE_RUN_DIR:-runs/tube_curriculum/bimanual_v1}"
cd "$PROJECT_ROOT"

{
    source "$PROJECT_ROOT/.venv-linux/bin/activate"
    export PATH="$PROJECT_ROOT/.venv-linux/bin:$PATH"
    export PYTHONUNBUFFERED=1

    command -v python
    python -c "import sys; print(\"Python:\", sys.version); print(\"Environment:\", sys.prefix)"

    bash "$PROJECT_ROOT/scripts/start_tube_curriculum.sh" \
        "$RUN_DIR" \
        --model "$MODEL_DIR"
} 2>&1 | tee -a "$PROJECT_ROOT/start_self.log"
'
