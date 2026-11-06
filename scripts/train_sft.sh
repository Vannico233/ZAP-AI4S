#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-linux/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" -m tube_agent.train_sft \
  --model "${TUBE_MODEL_PATH:-/inspire/hdd/project/robot-action/cengzhaoyu-25130105/wangningchong/DiG/assets/model}" "$@"
