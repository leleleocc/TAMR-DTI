#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"


# 2. 依次训练 5 个 ablation 实验
export SWANLAB_UPLOAD_TIMEOUT=30
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_CONFIG="configs/ds_zero2.json"
CACHE_ROOT="cache/features_n1"
SWANLAB_PROJECT="TAMR-DTI"
SWANLAB_LOG_ROOT="swanlog"

RUNS=(
  "datasets/biosnap|stage2-ablation-final-without-all-tamr-modules-biosnap-seed42"
  "datasets/biosnap|stage2-ablation-final-without-bidirectional-modulation-biosnap-seed42"
  "datasets/biosnap|stage2-ablation-final-without-ligand-conditioned-protein-film-biosnap-seed42"
  "datasets/biosnap|stage2-ablation-final-without-protein-mamba-refinement-biosnap-seed42"
  "datasets/biosnap|stage2-ablation-final-without-target-aware-conformer-module-biosnap-seed42"
)

for run_spec in "${RUNS[@]}"; do
  IFS='|' read -r dataset experiment_name <<< "${run_spec}"
  config_path="configs/experiments/${experiment_name}.yaml"
  save_dir="outputs/result/${experiment_name}"

  echo ""
  echo "=== Running: ${experiment_name} ==="
  scripts/run_deepspeed.sh "${dataset}" random \
    --num_gpus "${NUM_GPUS}" \
    --master_port "${MASTER_PORT}" \
    --deepspeed_config "${DEEPSPEED_CONFIG}" \
    --config "${config_path}" \
    --cache_root "${CACHE_ROOT}" \
    --save_dir "${save_dir}" \
    --swanlab_project "${SWANLAB_PROJECT}" \
    --swanlab_experiment "${experiment_name}" \
    --swanlab_log_root "${SWANLAB_LOG_ROOT}"
done

echo ""
echo "=== All 5 ablation experiments completed ==="
