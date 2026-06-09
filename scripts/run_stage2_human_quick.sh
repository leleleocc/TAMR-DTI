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
SPLIT="${SPLIT:-random}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-5}"
DRY_RUN="${DRY_RUN:-0}"

RUNS=(
  "01|stage2_human_best_01_weak_mamba_no_pos_weight_seed42|stage2-human-best-weak-mamba-no-pos-weight-seed42"
  "02|stage2_human_best_02_weak_mamba_no_pos_weight_seed43|stage2-human-best-weak-mamba-no-pos-weight-seed43"
  "03|stage2_human_best_03_weak_mamba_no_pos_weight_seed44|stage2-human-best-weak-mamba-no-pos-weight-seed44"
  "04|stage2_human_best_04_weak_mamba_no_pos_weight_seed45|stage2-human-best-weak-mamba-no-pos-weight-seed45"
  "05|stage2_human_best_05_weak_mamba_no_pos_weight_seed47|stage2-human-best-weak-mamba-no-pos-weight-seed47"
)

run_command() {
  local index="$1"
  local config_stem="$2"
  local experiment_name="$3"
  local config_path="configs/experiments/${config_stem}.yaml"
  local save_dir="outputs/result/${experiment_name}-${SPLIT}"

  local cmd=(
    scripts/run_deepspeed.sh datasets/human "${SPLIT}"
    --num_gpus "${NUM_GPUS}"
    --master_port "${MASTER_PORT}"
    --deepspeed_config "${DEEPSPEED_CONFIG}"
    --config "${config_path}"
    --cache_root "${CACHE_ROOT}"
    --save_dir "${save_dir}"
    --swanlab_project "${SWANLAB_PROJECT}"
    --swanlab_experiment "${experiment_name}-${SPLIT}"
    --swanlab_log_root "${SWANLAB_LOG_ROOT}"
  )

  echo "[stage2-human-quick ${index}] ${experiment_name}-${SPLIT}"
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
  run_command "${index}" "${config_stem}" "${experiment_name}"
done
