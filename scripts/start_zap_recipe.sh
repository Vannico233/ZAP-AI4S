#!/usr/bin/env bash
# Four GPUs: bounded warm-up, self-success search, SFT, RL, evaluation, videos.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT="${1:-runs/lab_zap_recipe/seed0_v5}"
if [[ $# -gt 0 ]]; then shift; fi
exec bash scripts/start_lab_all.sh "$OUTPUT" \
  --zap-recipe --action-protocol zap_pose_v1 \
  --sam-url http://127.0.0.1:8002 --follow "$@"
