#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please activate the conda env first, e.g. conda activate ldm-dti" >&2
  exit 1
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

print_usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/run_deepspeed.sh <dataset> <split> \
    --num_gpus <N> \
    --master_port <PORT> \
    --deepspeed_config <PATH> \
    --config <PATH> \
    --cache_root <PATH> \
    --save_dir <PATH> \
    --swanlab_project <NAME> \
    --swanlab_experiment <NAME> \
    --swanlab_log_root <PATH>

Example:
  scripts/run_deepspeed.sh datasets/bindingdb random \
    --num_gpus 8 \
    --master_port 29500 \
    --deepspeed_config configs/ds_zero2.json \
    --config configs/experiments/formal_e1_bimod_biintention_bindingdb_random.yaml \
    --cache_root cache/features \
    --save_dir outputs/result/formal-e1-bimod-biintention-bindingdb-random \
    --swanlab_project LDM-DTI \
    --swanlab_experiment formal-e1-bimod-biintention-bindingdb-random \
    --swanlab_log_root swanlog
USAGE
}

require_arg_value() {
  local key="$1"
  shift
  local args=("$@")
  local idx
  for idx in "${!args[@]}"; do
    local arg="${args[$idx]}"
    if [[ "${arg}" == "${key}="* ]]; then
      if [[ -n "${arg#*=}" ]]; then
        return 0
      fi
      echo "Missing value for required argument: ${key}" >&2
      return 1
    fi
    if [[ "${arg}" == "${key}" ]]; then
      local next_idx=$((idx + 1))
      if [[ "${next_idx}" -lt "${#args[@]}" && -n "${args[$next_idx]}" && "${args[$next_idx]}" != --* ]]; then
        return 0
      fi
      echo "Missing value for required argument: ${key}" >&2
      return 1
    fi
  done
  echo "Missing required explicit argument: ${key}" >&2
  return 1
}

if [[ $# -lt 2 ]]; then
  print_usage
  exit 1
fi

DATASET="$1" # 指定数据集
shift
SPLIT="$1" # 指定数据集分割
shift
if [[ "${SPLIT}" == --* ]]; then
  echo "Missing required explicit split argument before flags" >&2
  print_usage
  exit 1
fi

NUM_GPUS=""
MASTER_PORT=""
DS_CONFIG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_gpus)
      if [[ $# -lt 2 ]]; then
        echo "--num_gpus requires a value" >&2
        exit 1
      fi
      NUM_GPUS="$2"
      shift 2
      ;;
    --num_gpus=*)
      NUM_GPUS="${1#*=}"
      shift
      ;;
    --master_port)
      if [[ $# -lt 2 ]]; then
        echo "--master_port requires a value" >&2
        exit 1
      fi
      MASTER_PORT="$2"
      shift 2
      ;;
    --master_port=*)
      MASTER_PORT="${1#*=}"
      shift
      ;;
    --deepspeed_config)
      if [[ $# -lt 2 ]]; then
        echo "--deepspeed_config requires a value" >&2
        exit 1
      fi
      DS_CONFIG="$2"
      shift 2
      ;;
    --deepspeed_config=*)
      DS_CONFIG="${1#*=}"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

MISSING=false
if [[ -z "${NUM_GPUS}" ]]; then
  echo "Missing required explicit argument: --num_gpus" >&2
  MISSING=true
fi
if [[ -z "${MASTER_PORT}" ]]; then
  echo "Missing required explicit argument: --master_port" >&2
  MISSING=true
fi
if [[ -z "${DS_CONFIG}" ]]; then
  echo "Missing required explicit argument: --deepspeed_config" >&2
  MISSING=true
fi

for required in \
  --config \
  --cache_root \
  --save_dir \
  --swanlab_project \
  --swanlab_experiment \
  --swanlab_log_root; do
  if ! require_arg_value "${required}" "${EXTRA_ARGS[@]}"; then
    MISSING=true
  fi
done

if [[ "${MISSING}" == true ]]; then
  echo >&2
  print_usage
  exit 1
fi


deepspeed \
  --num_gpus="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  src/core/main_ds.py \
  --data "${DATASET}" \
  --split "${SPLIT}" \
  --deepspeed \
  --deepspeed_config "${DS_CONFIG}" \
  "${EXTRA_ARGS[@]}"
