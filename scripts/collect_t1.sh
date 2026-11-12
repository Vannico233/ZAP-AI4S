#!/usr/bin/env bash
# One bounded collection on the shared four-H100 notebook.
# Usage: bash scripts/collect_t1.sh MODEL_DIRECTORY OUTPUT_DIRECTORY [--parent PRIOR_BATCH] [options]
set -euo pipefail
if (( $# < 2 )); then
    echo "Usage: bash scripts/collect_t1.sh MODEL_DIRECTORY OUTPUT_DIRECTORY [collector options]" >&2
    exit 2
fi
model=$(realpath "$1")
out=$(realpath -m "$2")
shift 2
module=tube_agent.curriculum.collect_t1
options=()
while (( $# )); do
    case "$1" in
        --parent|--parent=*)
            if [[ "$1" == "--parent" ]]; then
                (( $# >= 2 )) || { echo "Missing --parent directory" >&2; exit 2; }
                shift
                parent=$(realpath "$1")
            else
                parent=$(realpath "${1#--parent=}")
            fi
            if [[ "$out/" == "$parent/"* || "$parent/" == "$out/"* ]]; then
                echo "The parent and continuation outputs require separate directories." >&2
                exit 2
            fi
            module=tube_agent.curriculum.continue_t1
            options+=(--parent "$parent")
            ;;
        *) options+=("$1") ;;
    esac
    shift
done
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ZAP_PROJECT_ROOT="$PWD" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 LP_NUM_THREADS=4
export ZAP_SOFTWARE_RENDER_GPUS="${ZAP_SOFTWARE_RENDER_GPUS:-2,3}"
mkdir -p "$out/operator"
exec 8>>"$out/operator/collection.lock"
if ! flock -n 8; then
    echo "Another collection is already active in $out" >&2
    exit 1
fi
logs=$(mktemp -d "$out/operator/launch-XXXXXXXX")
printf '%s\n' "$logs" > "$out/operator/latest"
git rev-parse HEAD > "$logs/revision"
python="$PWD/.venv-linux/bin/python"
port="${T1_SAM_PORT:-8002}"
url="http://127.0.0.1:$port"
weights="$PWD/assets/sam3.1/sam3.1_multiplex.pt"
sam_pid=
workers=()
cleanup() {
    for pid in "${workers[@]}"; do kill "$pid" 2>/dev/null || true; done
    if [[ -n "$sam_pid" ]]; then kill "$sam_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
if ! curl --noproxy '*' -fsS --max-time 4 "$url/health" -o "$logs/sam-health.json"; then
    CUDA_VISIBLE_DEVICES=1 "$PWD/.venv-sam/bin/python" -u -m tube_agent.lab.sam_server \
        --checkpoint "$weights" --port "$port" > "$logs/sam.log" 2>&1 &
    sam_pid=$!
    printf '%s\n' "$sam_pid" > "$logs/sam.pid"
    while ! curl --noproxy '*' -fsS --max-time 4 "$url/health" -o "$logs/sam-health.json" 2>/dev/null; do
        if ! kill -0 "$sam_pid" 2>/dev/null; then
            echo "SAM exited during startup. See $logs/sam.log" >&2
            exit 1
        fi
        sleep 3
    done
fi
"$python" - "$logs/sam-health.json" "$PWD/assets/sam3.1/SAM_MANIFEST.json" <<'PY'
import json, sys
from pathlib import Path
health, manifest = (json.loads(Path(name).read_text()) for name in sys.argv[1:])
if not (health.get("ready") and health.get("backend") == "sam3.1_detector"
        and health.get("checkpoint_sha256") == manifest["sha256"]):
    raise SystemExit("Unexpected SAM service or checkpoint")
print("SAM ready:", health["checkpoint_sha256"])
PY
gpus=(0 2 3)
for shard in 0 1 2; do
    CUDA_VISIBLE_DEVICES="${gpus[$shard]}" bash scripts/diagnostics/round10_egl/with_mapped_egl.sh \
        "$logs/render_audit" "$python" -u -m "$module" \
        --output "$out" --model "$model" --sam-url "$url" --shards 3 --shard "$shard" "${options[@]}" \
        > "$logs/shard_$shard.log" 2>&1 &
    workers+=("$!")
done
result=0
for pid in "${workers[@]}"; do wait "$pid" || result=1; done
workers=()
"$python" -m "$module" --output "$out" --summarize
exit "$result"
