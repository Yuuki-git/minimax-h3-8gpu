#!/bin/bash
set -euo pipefail
ss5_entry=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ss5_workspace=$(cd -P -- "${IFAB_WORKSPACE_ROOT:?Use the organizer web session workspace}" && pwd)
SS5_OPT_WORK=$(python3 -B - "$ss5_workspace" "${SS5_OPT_WORK:-$ss5_workspace/ss5-optimized}" <<'PY'
from pathlib import Path
import sys
workspace, work = map(lambda x: Path(x).resolve(), sys.argv[1:])
if not work.is_relative_to(workspace.parent) or work in (workspace, workspace.parent):
    raise SystemExit('Use a dedicated work directory inside the web workspace or its parent')
print(work)
PY
)
export SS5_OPT_WORK
mkdir -p -- "$SS5_OPT_WORK/tmp" "$SS5_OPT_WORK/logs" "$SS5_OPT_WORK/infini/lib"
export TMPDIR="$SS5_OPT_WORK/tmp" TMP="$SS5_OPT_WORK/tmp" TEMP="$SS5_OPT_WORK/tmp"
set +u
source /opt/dtk/env.sh
set -u
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export INFINI_ROOT="$SS5_OPT_WORK/infini" XDG_CACHE_HOME="$SS5_OPT_WORK/cache"
export MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 XMAKE_ROOT=y XMAKE_STATS=false
export XMAKE_GLOBALDIR="$SS5_OPT_WORK/xmake-global" XMAKE_PKG_CACHEDIR="$SS5_OPT_WORK/xmake-cache"
export XMAKE_PKG_INSTALLDIR="$SS5_OPT_WORK/xmake-packages" XMAKE_TMPDIR="$TMPDIR" XMAKE_RAMDIR="$TMPDIR"
for ss5_name in ninetoothed triton triton-dump triton-override cuda torch torch-extensions miopen-db miopen hf/hub matplotlib pip; do
  mkdir -p -- "$XDG_CACHE_HOME/$ss5_name"
done
export NINETOOTHED_CACHE_DIR="$XDG_CACHE_HOME/ninetoothed" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TRITON_DUMP_DIR="$XDG_CACHE_HOME/triton-dump" TRITON_OVERRIDE_DIR="$XDG_CACHE_HOME/triton-override"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export MIOPEN_USER_DB_PATH="$XDG_CACHE_HOME/miopen-db" MIOPEN_CUSTOM_CACHE_DIR="$XDG_CACHE_HOME/miopen"
export HF_HOME="$XDG_CACHE_HOME/hf" HF_HUB_CACHE="$XDG_CACHE_HOME/hf/hub"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export IFAB_SDK_LOG_ROOT="$SS5_OPT_WORK/logs" SS5_ROUND1_CACHE_SCOPE="$XDG_CACHE_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL="$SS5_OPT_WORK/gitconfig"
[ -e "$GIT_CONFIG_GLOBAL" ] || : > "$GIT_CONFIG_GLOBAL"
export SS5_BASE_PYTHON="$SS5_OPT_WORK/venv/bin/python"
export PATH="$SS5_OPT_WORK/deps/tools/bin:$PATH"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/usr/local/lib/python3.11/site-packages/torch.libs:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SS5_OPT_WORK/src/InfiniCore/python:$SS5_OPT_WORK/src/InfiniLM/python"
export SS5_NT_ATTN="${SS5_NT_ATTN:-1}" SS5_NT_ADALN="${SS5_NT_ADALN:-1}"
case "$SS5_NT_ATTN$SS5_NT_ADALN" in 00|01|10|11) ;; *) echo 'Switches must be 00, 01, 10 or 11' >&2; return 1 ;; esac
if [ -x "$SS5_BASE_PYTHON" ]; then
  export FA_SO_PATH
  FA_SO_PATH=$("$SS5_BASE_PYTHON" -B - <<'PY'
from pathlib import Path
import sys
matches = [p for root in sys.path if root and ('site-packages' in root or 'dist-packages' in root)
           for p in Path(root).glob('flash_attn_2_cuda*.so')]
if not matches:
    raise SystemExit('The organizer vendor FlashAttention extension is required')
print(matches[0].resolve())
PY
)
fi
