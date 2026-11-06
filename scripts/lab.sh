#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-linux/bin/python"
if [[ ! -x "$PY" ]]; then echo "Missing .venv-linux. Run scripts/prepare_offline.sh first." >&2; exit 1; fi
export MUJOCO_GL="${TUBE_MUJOCO_GL:-egl}"
if [[ "$MUJOCO_GL" == egl && -f "$ROOT/.runtime/egl-jammy/EGL_RUNTIME.json" ]]; then
  "$PY" scripts/check_render.py --verify-runtime
  export LD_LIBRARY_PATH="$ROOT/.runtime/egl-jammy/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_$$"
STAGE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$STAGE" in
  prepare)
    bash scripts/prepare_sam_offline.sh "$@" ;;
  sam)
    if [[ ! -x .venv-sam/bin/python ]]; then echo "Run: bash scripts/lab.sh prepare" >&2; exit 1; fi
    export CUDA_VISIBLE_DEVICES="${SAM_GPU:-1}"
    exec .venv-sam/bin/python -m tube_agent.lab.sam_server --checkpoint "${TUBE_SAM_CHECKPOINT:-$ROOT/assets/sam3.1/sam3.1_multiplex.pt}" "$@" ;;
  model)
    export CUDA_VISIBLE_DEVICES="${QWEN_GPU:-0}"
    exec "$PY" -m tube_agent.serve --max-context 16384 "$@" ;;
  physics)
    export MUJOCO_GL=disable
    exec "$PY" -m tube_agent.lab.doctor --physics-only --output "runs/lab_v1/physics_$STAMP" "$@" ;;
  check)
    exec "$PY" -m tube_agent.lab.doctor --services --output "runs/lab_v1/preflight_$STAMP" "$@" ;;
  render-check)
    exec "$PY" scripts/check_render.py --output "runs/lab_v1/render_$STAMP" "$@" ;;
  sample|search|eval)
    EXTRA=()
    if [[ "$STAGE" == search ]]; then EXTRA+=(--search); ATTEMPTS=32; SPLIT=train
    elif [[ "$STAGE" == eval ]]; then ATTEMPTS=4; SPLIT=test
    else ATTEMPTS=8; SPLIT=train; fi
    for TASK in pick transport insert unscrew; do
      "$PY" -m tube_agent.lab.run --task "$TASK" --split "$SPLIT" --seed 0 --attempts "$ATTEMPTS" \
        --output "runs/lab_v1/$STAGE" "${EXTRA[@]}" "$@"
    done
    if [[ "$STAGE" == eval ]]; then
      "$PY" -m tube_agent.lab.run --task rack_transfer --split test --seed 0 --attempts "$ATTEMPTS" \
        --output runs/lab_v1/eval_generalization "$@"
    fi ;;
  export)
    exec "$PY" -m tube_agent.lab.train export --source runs/lab_v1 --output "data/lab_visual_success_$STAMP.jsonl" "$@" ;;
  sft|grpo)
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-2}"
    exec "$PY" -m tube_agent.lab.train "$STAGE" --output "checkpoints/lab_${STAGE}_$STAMP" "$@" ;;
  replay)
    exec "$PY" -m tube_agent.lab.replay "$@" ;;
  *)
    echo "Usage: bash scripts/lab.sh {prepare|sam|model|physics|render-check|check|sample|search|export|sft|grpo|eval|replay} [options]"
    echo "sam: GPU 1; model: GPU 0; sft/grpo: GPU 2. Override SAM_GPU, QWEN_GPU, TRAIN_GPU."
    echo "First run physics, then start sam + model in separate terminals, then check."
    [[ "$STAGE" == help ]] || exit 1 ;;
esac
