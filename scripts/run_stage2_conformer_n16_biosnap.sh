#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29516}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_zero2.json}"
CACHE_ROOT="${CACHE_ROOT:-cache/features_n16}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-LDM-DTI}"
SWANLAB_LOG_ROOT="${SWANLAB_LOG_ROOT:-swanlog}"
DRY_RUN="${DRY_RUN:-0}"

DATASET="datasets/biosnap"
SPLIT="random"
CONFIG_PATH="configs/experiments/stage2-comparative-conformer-n16-biosnap-seed42.yaml"
EXPERIMENT_NAME="stage2-comparative-conformer-n16-biosnap-seed42"
SAVE_DIR="outputs/result/${EXPERIMENT_NAME}"

cmd=(
  scripts/run_deepspeed.sh "${DATASET}" "${SPLIT}"
  --num_gpus "${NUM_GPUS}"
  --master_port "${MASTER_PORT}"
  --deepspeed_config "${DEEPSPEED_CONFIG}"
  --config "${CONFIG_PATH}"
  --cache_root "${CACHE_ROOT}"
  --save_dir "${SAVE_DIR}"
  --swanlab_project "${SWANLAB_PROJECT}"
  --swanlab_experiment "${EXPERIMENT_NAME}"
  --swanlab_log_root "${SWANLAB_LOG_ROOT}"
)

echo "[stage2-conformer-n16] ${EXPERIMENT_NAME}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '  '
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
