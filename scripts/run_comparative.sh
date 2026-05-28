#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export SWANLAB_UPLOAD_TIMEOUT=30
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_CONFIG="configs/ds_zero2.json"
SWANLAB_PROJECT="TAMR-DTI"
SWANLAB_LOG_ROOT="swanlog"

# format: cache_root|dataset|experiment_name
RUNS=(
  "cache/features_n1|datasets/biosnap|stage2-comparative-conformer-n1-biosnap-seed42"
  "cache/features_n2|datasets/biosnap|stage2-comparative-conformer-n2-biosnap-seed42"
  "cache/features_n4|datasets/biosnap|stage2-comparative-conformer-n4-biosnap-seed42"
  "cache/features|datasets/biosnap|stage2-comparative-conformer-n8-biosnap-seed42"
  "cache/features|datasets/biosnap|stage2-comparative-conformer-avg-biosnap-seed42"
)

for run_spec in "${RUNS[@]}"; do
  IFS='|' read -r cache_root dataset experiment_name <<< "${run_spec}"
  config_path="configs/experiments/${experiment_name}.yaml"
  save_dir="outputs/result/${experiment_name}"

  echo ""
  echo "=== Running: ${experiment_name} ==="
  scripts/run_deepspeed.sh "${dataset}" random \
    --num_gpus "${NUM_GPUS}" \
    --master_port "${MASTER_PORT}" \
    --deepspeed_config "${DEEPSPEED_CONFIG}" \
    --config "${config_path}" \
    --cache_root "${cache_root}" \
    --save_dir "${save_dir}" \
    --swanlab_project "${SWANLAB_PROJECT}" \
    --swanlab_experiment "${experiment_name}" \
    --swanlab_log_root "${SWANLAB_LOG_ROOT}"
done

echo ""
echo "=== All 5 comparative experiments completed ==="
