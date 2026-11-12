#!/usr/bin/env bash
# Fixed 960-episode warm-up, shared 128-continuation search, SFT, three-node eval.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="${1:-runs/tube_curriculum/bimanual_v1}"
if [[ $# -gt 0 ]]; then shift; fi
PY="$ROOT/.venv-linux/bin/python"
if [[ ! -x "$PY" ]]; then echo "Missing existing .venv-linux/bin/python" >&2; exit 1; fi
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
STARTED_AT="$(date +%s)"
nohup bash scripts/lab.sh tube-curriculum --output "$OUTPUT" "$@" >> "$OUTPUT/launcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUTPUT/launcher.pid"
echo "Curriculum PID: $PID"
echo "Progress: $OUTPUT/status.json"
echo "Results: $OUTPUT/report.json"
echo "Same command resumes committed work. Ctrl+C leaves background work running."
exec "$PY" scripts/watch_lab_run.py "$OUTPUT" --pid "$PID" --since "$STARTED_AT"
