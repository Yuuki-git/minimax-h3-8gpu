#!/bin/bash
set -euo pipefail
here=$(cd -P -- "$(dirname -- "$0")" && pwd)
source "$here/optimized_env.sh"
[ -r "$SS5_OPT_WORK/source-lock.json" ] || { echo 'Run fetch_sources.sh first' >&2; exit 1; }
cd -- "$ss5_workspace"
bash "$here/optimized_core_build.sh" inputs
"$SS5_BASE_PYTHON" -B "$here/prepare_codegen_env.py"
mkdir -p -- "$SS5_OPT_WORK/codegen"
export TRITON_CACHE_MANAGER=round1_triton_cache:WorkspaceCacheManager
for kernel in adaln gate attention; do
  destination=$(mktemp -d -- "$SS5_OPT_WORK/codegen/$kernel.XXXXXXXX")
  PYTHONPATH="$ss5_entry/codegen:$PYTHONPATH" "$SS5_OPT_WORK/codegen-venv/bin/python" -B \
    "$ss5_entry/codegen/generate_native.py" --case "$kernel" --output "$destination" --prefix "$INFINI_ROOT" \
    2>&1 | tee "$destination/generation.log"
done
unset TRITON_CACHE_MANAGER
bash "$here/optimized_core_build.sh" core-mha
bash "$here/optimized_core_build.sh" lm
"$SS5_BASE_PYTHON" -B "$here/record_optimized_build.py"
echo "SS5_OPTIMIZED_BUILD_OK=$SS5_OPT_WORK"
