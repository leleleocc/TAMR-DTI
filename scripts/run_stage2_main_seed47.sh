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
START_INDEX="${START_INDEX:-21}"
END_INDEX="${END_INDEX:-24}"
DRY_RUN="${DRY_RUN:-0}"

RUNS=(
  "21|datasets/bindingdb|stage2_main_21_full_tamr_dti_n8_bindingdb_seed47|stage2-main-21-full-tamr-dti-n8-bindingdb-seed47"
  "22|datasets/biosnap|stage2_main_22_full_tamr_dti_n8_biosnap_seed47|stage2-main-22-full-tamr-dti-n8-biosnap-seed47"
  "23|datasets/human|stage2_main_23_full_tamr_dti_n8_human_seed47|stage2-main-23-full-tamr-dti-n8-human-seed47"
  "24|datasets/celegans|stage2_main_24_full_tamr_dti_n8_celegans_seed47|stage2-main-24-full-tamr-dti-n8-celegans-seed47"
)

run_command() {
  local index="$1"
  local dataset="$2"
  local config_stem="$3"
  local experiment_name="$4"
  local config_path="configs/experiments/${config_stem}.yaml"
  local save_dir="outputs/result/${experiment_name}"

  local cmd=(
    scripts/run_deepspeed.sh "${dataset}" random
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

  echo "[stage2-main-seed47 ${index}] ${experiment_name}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '  '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

for run_spec in "${RUNS[@]}"; do
  IFS='|' read -r index dataset config_stem experiment_name <<< "${run_spec}"
  index_num=$((10#${index}))
  if (( index_num < START_INDEX || index_num > END_INDEX )); then
    continue
  fi
  run_command "${index}" "${dataset}" "${config_stem}" "${experiment_name}"
done
