#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-datasets/biosnap}"
SPLIT="${SPLIT:-random}"
CACHE_ROOT="${CACHE_ROOT:-cache/features_n16}"
BASE_CACHE_ROOT="${BASE_CACHE_ROOT:-cache/features}"
DEVICE="${DEVICE:-cpu}"
NUM_CONFORMERS="${NUM_CONFORMERS:-16}"
DRUG3D_WORKERS="${DRUG3D_WORKERS:-8}"
DRUG3D_SAVE_EVERY="${DRUG3D_SAVE_EVERY:-250}"
DRUG3D_TIMEOUT="${DRUG3D_TIMEOUT:-60}"
OVERWRITE_DRUG3D="${OVERWRITE_DRUG3D:-1}"
DRY_RUN="${DRY_RUN:-0}"

NVJITLINK_DIR="${NVJITLINK_DIR:-/home/lsw/miniconda3/envs/ldm-dti/lib/python3.9/site-packages/nvidia/nvjitlink/lib}"
if [[ -d "${NVJITLINK_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NVJITLINK_DIR}:${LD_LIBRARY_PATH:-}"
fi

cache_dir="${CACHE_ROOT}/${DATASET}/${SPLIT}"
base_cache_dir="${BASE_CACHE_ROOT}/${DATASET}/${SPLIT}"
mkdir -p "${cache_dir}"

for name in smiles_features.pt protein_features.pt; do
  if [[ ! -e "${cache_dir}/${name}" && ! -L "${cache_dir}/${name}" && -e "${base_cache_dir}/${name}" ]]; then
    ln -s "${ROOT_DIR}/${base_cache_dir}/${name}" "${cache_dir}/${name}"
  fi
done

cmd=(
  "${PYTHON_BIN}" scripts/pre_extract.py
  --data "${DATASET}"
  --split "${SPLIT}"
  --device "${DEVICE}"
  --output_dir "${CACHE_ROOT}"
  --num_conformers "${NUM_CONFORMERS}"
  --drug3d_workers "${DRUG3D_WORKERS}"
  --drug3d_save_every "${DRUG3D_SAVE_EVERY}"
  --drug3d_timeout "${DRUG3D_TIMEOUT}"
)

if [[ "${OVERWRITE_DRUG3D}" == "1" ]]; then
  cmd+=(--overwrite_drug3d)
fi

echo "[pre-extract-biosnap-n16] ${DATASET}/${SPLIT} -> ${CACHE_ROOT}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '  '
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
