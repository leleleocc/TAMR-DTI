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
END_INDEX="${END_INDEX:-6}"
DRY_RUN="${DRY_RUN:-0}"

RUNS=(
  "01|datasets/bindingdb|stage2_ablation_01_without_target_aware_conformer_selection_bindingdb_seed42|stage2-ablation-01-without-target-aware-conformer-selection-bindingdb-seed42"
  "02|datasets/bindingdb|stage2_ablation_02_without_ligand_conditioned_protein_film_bindingdb_seed42|stage2-ablation-02-without-ligand-conditioned-protein-film-bindingdb-seed42"
  "03|datasets/bindingdb|stage2_ablation_03_without_protein_mamba_refinement_bindingdb_seed42|stage2-ablation-03-without-protein-mamba-refinement-bindingdb-seed42"
  "04|datasets/biosnap|stage2_ablation_04_without_target_aware_conformer_selection_biosnap_seed42|stage2-ablation-04-without-target-aware-conformer-selection-biosnap-seed42"
  "05|datasets/biosnap|stage2_ablation_05_without_ligand_conditioned_protein_film_biosnap_seed42|stage2-ablation-05-without-ligand-conditioned-protein-film-biosnap-seed42"
  "06|datasets/biosnap|stage2_ablation_06_without_protein_mamba_refinement_biosnap_seed42|stage2-ablation-06-without-protein-mamba-refinement-biosnap-seed42"
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

  echo "[stage2-ablation ${index}] ${experiment_name}"
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
