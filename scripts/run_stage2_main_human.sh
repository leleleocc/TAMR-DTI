#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_zero2.json}"
CACHE_ROOT="${CACHE_ROOT:-cache/features}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-LDM-DTI}"
SWANLAB_LOG_ROOT="${SWANLAB_LOG_ROOT:-swanlog}"
SPLIT="${SPLIT:-random2}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-5}"
DRY_RUN="${DRY_RUN:-0}"

RUNS=(
  "01|datasets/human|stage2_main_11_full_tamr_dti_n8_human_seed42|42"
  "02|datasets/human|stage2_main_12_full_tamr_dti_n8_human_seed43|43"
  "03|datasets/human|stage2_main_13_full_tamr_dti_n8_human_seed44|44"
  "04|datasets/human|stage2_main_14_full_tamr_dti_n8_human_seed45|45"
  "05|datasets/human|stage2_main_23_full_tamr_dti_n8_human_seed47|47"
)

run_command() {
  local index="$1"
  local dataset="$2"
  local config_stem="$3"
  local seed="$4"
  local experiment_name="stage2-main-human-${SPLIT}-full-tamr-dti-n8-seed${seed}"
  local config_path="configs/experiments/${config_stem}.yaml"
  local save_dir="outputs/result/${experiment_name}"

  local cmd=(
    scripts/run_deepspeed.sh "${dataset}" "${SPLIT}"
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

  echo "[stage2-main-human ${index}] ${experiment_name} split=${SPLIT}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

for run_spec in "${RUNS[@]}"; do
  IFS='|' read -r index dataset config_stem seed <<< "${run_spec}"
  index_num=$((10#${index}))
  if (( index_num < START_INDEX || index_num > END_INDEX )); then
    continue
  fi
  run_command "${index}" "${dataset}" "${config_stem}" "${seed}"
done
