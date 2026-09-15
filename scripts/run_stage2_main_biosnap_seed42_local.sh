#!/usr/bin/env bash
# Train BioSNAP seed-42 full TAMR-DTI (n=8 conformers) on the local /paddle box.
# - 8 x V100-SXM2-32G
# - conda env: ldm-dti at /paddle/miniconda3/envs/ldm-dti
# - swanlab offline (no login required, backup written under swanlog/)
# - checkpoint dir matches INTERPRETABILITY_PLAN expectations.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# ---- env -------------------------------------------------------------------
CONDA_BASE="${CONDA_BASE:-/paddle/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ldm-dti}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"

# ---- run knobs -------------------------------------------------------------
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_zero2.json}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiments/stage2_main_06_full_tamr_dti_n8_biosnap_seed42.yaml}"
CACHE_ROOT="${CACHE_ROOT:-cache/features}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-stage2-main-06-full-tamr-dti-n8-biosnap-seed42}"
SAVE_DIR="${SAVE_DIR:-outputs/result/${EXPERIMENT_NAME}}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-LDM-DTI}"
SWANLAB_LOG_ROOT="${SWANLAB_LOG_ROOT:-swanlog}"

mkdir -p "${SAVE_DIR}" "${SWANLAB_LOG_ROOT}" logs

LOG_FILE="${LOG_FILE:-logs/${EXPERIMENT_NAME}.log}"

echo "[run] $(date -Is) starting ${EXPERIMENT_NAME}"
echo "[run] save_dir=${SAVE_DIR}"
echo "[run] log_file=${LOG_FILE}"

deepspeed \
  --num_gpus="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  src/core/main_ds.py \
  --data datasets/biosnap \
  --split random \
  --deepspeed \
  --deepspeed_config "${DEEPSPEED_CONFIG}" \
  --config "${CONFIG_PATH}" \
  --cache_root "${CACHE_ROOT}" \
  --save_dir "${SAVE_DIR}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  --swanlab_experiment "${EXPERIMENT_NAME}" \
  --swanlab_log_root "${SWANLAB_LOG_ROOT}" \
  2>&1 | tee "${LOG_FILE}"
