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
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-20}"
DRY_RUN="${DRY_RUN:-0}"

RUNS=(
  "01|datasets/bindingdb|stage2_main_01_full_tamr_dti_n8_bindingdb_seed42|stage2-main-01-full-tamr-dti-n8-bindingdb-seed42"
  "02|datasets/bindingdb|stage2_main_02_full_tamr_dti_n8_bindingdb_seed43|stage2-main-02-full-tamr-dti-n8-bindingdb-seed43"
  "03|datasets/bindingdb|stage2_main_03_full_tamr_dti_n8_bindingdb_seed44|stage2-main-03-full-tamr-dti-n8-bindingdb-seed44"
  "04|datasets/bindingdb|stage2_main_04_full_tamr_dti_n8_bindingdb_seed45|stage2-main-04-full-tamr-dti-n8-bindingdb-seed45"
  "05|datasets/bindingdb|stage2_main_05_full_tamr_dti_n8_bindingdb_seed46|stage2-main-05-full-tamr-dti-n8-bindingdb-seed46"
  "06|datasets/biosnap|stage2_main_06_full_tamr_dti_n8_biosnap_seed42|stage2-main-06-full-tamr-dti-n8-biosnap-seed42"
  "07|datasets/biosnap|stage2_main_07_full_tamr_dti_n8_biosnap_seed43|stage2-main-07-full-tamr-dti-n8-biosnap-seed43"
  "08|datasets/biosnap|stage2_main_08_full_tamr_dti_n8_biosnap_seed44|stage2-main-08-full-tamr-dti-n8-biosnap-seed44"
  "09|datasets/biosnap|stage2_main_09_full_tamr_dti_n8_biosnap_seed45|stage2-main-09-full-tamr-dti-n8-biosnap-seed45"
  "10|datasets/biosnap|stage2_main_10_full_tamr_dti_n8_biosnap_seed46|stage2-main-10-full-tamr-dti-n8-biosnap-seed46"
  "11|datasets/human|stage2_main_11_full_tamr_dti_n8_human_seed42|stage2-main-11-full-tamr-dti-n8-human-seed42"
  "12|datasets/human|stage2_main_12_full_tamr_dti_n8_human_seed43|stage2-main-12-full-tamr-dti-n8-human-seed43"
  "13|datasets/human|stage2_main_13_full_tamr_dti_n8_human_seed44|stage2-main-13-full-tamr-dti-n8-human-seed44"
  "14|datasets/human|stage2_main_14_full_tamr_dti_n8_human_seed45|stage2-main-14-full-tamr-dti-n8-human-seed45"
  "15|datasets/human|stage2_main_15_full_tamr_dti_n8_human_seed46|stage2-main-15-full-tamr-dti-n8-human-seed46"
  "16|datasets/celegans|stage2_main_16_full_tamr_dti_n8_celegans_seed42|stage2-main-16-full-tamr-dti-n8-celegans-seed42"
  "17|datasets/celegans|stage2_main_17_full_tamr_dti_n8_celegans_seed43|stage2-main-17-full-tamr-dti-n8-celegans-seed43"
  "18|datasets/celegans|stage2_main_18_full_tamr_dti_n8_celegans_seed44|stage2-main-18-full-tamr-dti-n8-celegans-seed44"
  "19|datasets/celegans|stage2_main_19_full_tamr_dti_n8_celegans_seed45|stage2-main-19-full-tamr-dti-n8-celegans-seed45"
  "20|datasets/celegans|stage2_main_20_full_tamr_dti_n8_celegans_seed46|stage2-main-20-full-tamr-dti-n8-celegans-seed46"
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

  echo "[stage2-main ${index}] ${experiment_name}"
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
