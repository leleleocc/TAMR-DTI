#!/usr/bin/env bash
set -uo pipefail

# Stage 2 main training script for Human dataset with random3 split.
# Usage:
#   ./scripts/run_stage2_human_random3.sh
#
# Environment variables:
#   NUM_GPUS          - Number of GPUs (default: 8)
#   MASTER_PORT       - DeepSpeed master port (default: 29500)
#   DEEPSPEED_CONFIG  - DeepSpeed config path (default: configs/ds_zero2.json)
#   CACHE_ROOT        - Feature cache root (default: cache/features)
#   SWANLAB_PROJECT   - SwanLab project name (default: LDM-DTI)
#   SWANLAB_LOG_ROOT  - SwanLab log root (default: swanlog)
#   START_INDEX       - Start index of runs (default: 1)
#   END_INDEX         - End index of runs (default: 5)
#   DRY_RUN           - Set to 1 to print commands without executing (default: 0)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_zero2.json}"
CACHE_ROOT="${CACHE_ROOT:-cache/features}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-LDM-DTI}"
SWANLAB_LOG_ROOT="${SWANLAB_LOG_ROOT:-swanlog}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-5}"
DRY_RUN="${DRY_RUN:-0}"

# Format: index|config_stem|experiment_name
# Original seeds (commented out):
#RUNS=(
#  "1|stage2_main_human_random3_seed42|stage2-main-human-random3-seed42"
#  "2|stage2_main_human_random3_seed43|stage2-main-human-random3-seed43"
#  "3|stage2_main_human_random3_seed44|stage2-main-human-random3-seed44"
#  "4|stage2_main_human_random3_seed45|stage2-main-human-random3-seed45"
#  "5|stage2_main_human_random3_seed47|stage2-main-human-random3-seed47"
#)

# New seeds:
RUNS=(
  "1|stage2_main_human_random3_seed43|stage2-main-human-random3-seed43"
  "2|stage2_main_human_random3_seed48|stage2-main-human-random3-seed48"
  "3|stage2_main_human_random3_seed49|stage2-main-human-random3-seed49"
  "4|stage2_main_human_random3_seed52|stage2-main-human-random3-seed52"
  "5|stage2_main_human_random3_seed100|stage2-main-human-random3-seed100"
)

run_command() {
  local index="$1"
  local config_stem="$2"
  local experiment_name="$3"
  local config_path="configs/experiments/${config_stem}.yaml"
  local save_dir="outputs/result/${experiment_name}"

  local cmd=(
    scripts/run_deepspeed.sh datasets/human random3
    --num_gpus "${NUM_GPUS}"
    --master_port "${MASTER_PORT}"
    --deepspeed_config "${DEEPSPEED_CONFIG}"
    --config "${config_path}"
    --cache_root "${CACHE_ROOT}"
    --save_dir "${save_dir}"
    --swanlab_project "${SWANLAB_PROJECT}"
    --swanlab_experiment "${experiment_name}"
    --swanlab_log_root "${SWANLAB_LOG_ROOT}"
  )

  echo "[stage2-human-random3 ${index}] ${experiment_name}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

for run_spec in "${RUNS[@]}"; do
  IFS='|' read -r index config_stem experiment_name <<< "${run_spec}"
  index_num=$((10#${index}))
  if (( index_num < START_INDEX || index_num > END_INDEX )); then
    continue
  fi
  run_command "${index}" "${config_stem}" "${experiment_name}" || echo "[WARNING] Run ${index} (${experiment_name}) exited with non-zero status, continuing..."
done
