#!/usr/bin/env bash
# Disconnect-safe launcher. Same output + same options resumes committed work.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="${1:-runs/lab_success_first/fixed_seed0_v3}"
if [[ $# -gt 0 ]]; then shift; fi
FOLLOW=0
ARGS=()
for ARG in "$@"; do
  if [[ "$ARG" == --follow ]]; then FOLLOW=1; else ARGS+=("$ARG"); fi
done
if [[ "$FOLLOW" == 1 && ! -x "$ROOT/.venv-linux/bin/python" ]]; then
  echo "Missing .venv-linux/bin/python; prepare the environment first." >&2
  exit 1
fi
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
STARTED_AT="$(date +%s)"
nohup bash scripts/lab.sh all --output "$OUTPUT" "${ARGS[@]}" >> "$OUTPUT/launcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUTPUT/launcher.pid"
echo "Pipeline PID: $PID"
echo "Progress: $OUTPUT/status.json"
echo "Log: $OUTPUT/launcher.log"
echo "Final results: $OUTPUT/report.json"
echo "Repeat this command with the same output and options to resume after interruption."
if [[ "$FOLLOW" == 1 ]]; then
  exec "$ROOT/.venv-linux/bin/python" scripts/watch_lab_run.py "$OUTPUT" --pid "$PID" --since "$STARTED_AT"
fi
