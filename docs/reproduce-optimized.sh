#!/bin/bash
set -euo pipefail
cd -- "${IFAB_WORKSPACE_ROOT:?}"
export SS5_OPT_WORK="${IFAB_WORKSPACE_ROOT:?}/ss5-optimized"
git clone --branch v2-ss5-optimized --depth 1 \
  https://github.com/Yuuki-git/minimax-h3-8gpu.git
cd minimax-h3-8gpu
bash scripts/fetch_sources.sh
bash scripts/build_optimized.sh
bash scripts/run_optimized.sh smoke
bash scripts/run_optimized.sh reference \
  --output-dir "$SS5_OPT_WORK/results/reference"
bash scripts/run_optimized.sh native \
  --output-dir "$SS5_OPT_WORK/results/optimized"
bash scripts/compare.sh \
  --reference-video "$SS5_OPT_WORK/results/reference/reference_fl2va.mp4" \
  --native-video "$SS5_OPT_WORK/results/optimized/infinilm_fl2va.mp4" \
  --output-dir "$SS5_OPT_WORK/results/comparison"
