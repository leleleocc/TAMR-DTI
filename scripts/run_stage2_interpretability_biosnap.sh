#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXTRACT_PY="${EXTRACT_PY:-/home/lsw/miniconda3/envs/ldm-dti/bin/python}"
PLOT_PY="${PLOT_PY:-/home/lsw/miniconda3/bin/python}"

export LD_LIBRARY_PATH="/home/lsw/miniconda3/envs/ldm-dti/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

"$EXTRACT_PY" scripts/extract_stage2_interpretability_biosnap.py "$@"
"$PLOT_PY" scripts/plot_stage2_interpretability_biosnap.py
