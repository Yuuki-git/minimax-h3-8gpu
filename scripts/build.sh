#!/bin/bash
set -euo pipefail
here=$(cd -P -- "$(dirname -- "$0")" && pwd)
source "$here/env.sh"
export TRITON_CACHE_MANAGER=triton_cache:WorkspaceCacheManager
mkdir -p -- "$SS5_OPT_WORK/codegen"
for kernel in adaln gate attention; do
  destination=$(mktemp -d -- "$SS5_OPT_WORK/codegen/$kernel.XXXXXXXX")
  PYTHONPATH="$ss5_entry/codegen:$PYTHONPATH" "$SS5_OPT_WORK/codegen-venv/bin/python" -B \
    "$ss5_entry/codegen/generate.py" --case "$kernel" --output "$destination" --prefix "$INFINI_ROOT" \
    2>&1 | tee "$destination/build.log"
done
unset TRITON_CACHE_MANAGER
bash "$here/build_native.sh"
touch "$SS5_OPT_WORK/.built"
echo "BUILD_OK=$SS5_OPT_WORK"
