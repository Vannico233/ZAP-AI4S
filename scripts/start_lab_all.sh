#!/usr/bin/env bash
# Disconnect-safe launcher. Same output + same options resumes committed work.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="${1:-runs/lab_success_first/fixed_seed0_v2}"
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
nohup bash scripts/lab.sh all --output "$OUTPUT" "$@" >> "$OUTPUT/launcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUTPUT/launcher.pid"
echo "Pipeline PID: $PID"
echo "Progress: $OUTPUT/status.json"
echo "Log: $OUTPUT/launcher.log"
echo "Final results: $OUTPUT/report.json"
echo "Repeat this command with the same output and options to resume after interruption."
