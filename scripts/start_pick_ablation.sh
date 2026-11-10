#!/usr/bin/env bash
# Existing offline environment; bounded pick-only experiment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="${1:-runs/lab_pick_ablation/v1}"
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
STARTED_AT="$(date +%s)"
nohup bash scripts/lab.sh pick-ablation --output "$OUTPUT" "$@" >> "$OUTPUT/launcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUTPUT/launcher.pid"
echo "Pick ablation PID: $PID"
echo "Results: $OUTPUT/report.json"
exec "$ROOT/.venv-linux/bin/python" scripts/watch_lab_run.py "$OUTPUT" --pid "$PID" --since "$STARTED_AT"
