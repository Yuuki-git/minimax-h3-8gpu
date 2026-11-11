#!/bin/bash
# Source this file; every writable path is inside the web workspace or its parent.
set -euo pipefail
ss5_entry=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ss5_workspace=$(cd -P -- "${IFAB_WORKSPACE_ROOT:?Run in the platform web workspace}" && pwd)
ss5_afs=$(cd -P -- "${IFAB_AFS_HOME:?Read-only platform inputs are required}" && pwd)
SS5_OPT_WORK=$(python3 -B - "$ss5_workspace" "$ss5_afs" "${SS5_OPT_WORK:-$ss5_workspace/ss5-optimized}" <<'PY'
from pathlib import Path
import sys
w, afs, target = map(lambda p: Path(p).resolve(), sys.argv[1:])
if not target.is_relative_to(w.parent) or target in (w.parent, w, afs) or target.is_relative_to(afs):
    raise SystemExit('SS5_OPT_WORK must be a dedicated directory within the web workspace or its parent')
print(target)
PY
)
export SS5_OPT_WORK
mkdir -p -- "$SS5_OPT_WORK" "$SS5_OPT_WORK/tmp" "$SS5_OPT_WORK/logs" "$SS5_OPT_WORK/infini/lib"
export TMPDIR="$SS5_OPT_WORK/tmp" TMP="$SS5_OPT_WORK/tmp" TEMP="$SS5_OPT_WORK/tmp"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 XMAKE_STATS=false
export INFINI_ROOT="$SS5_OPT_WORK/infini"
export XDG_CACHE_HOME="$SS5_OPT_WORK/cache"
for ss5_cache in ninetoothed triton triton-dump triton-override cuda torch torch-extensions miopen-db miopen hf/hub matplotlib pip; do
  mkdir -p -- "$XDG_CACHE_HOME/$ss5_cache"
done
export NINETOOTHED_CACHE_DIR="$XDG_CACHE_HOME/ninetoothed"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TRITON_DUMP_DIR="$XDG_CACHE_HOME/triton-dump"
export TRITON_OVERRIDE_DIR="$XDG_CACHE_HOME/triton-override" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export MIOPEN_USER_DB_PATH="$XDG_CACHE_HOME/miopen-db" MIOPEN_CUSTOM_CACHE_DIR="$XDG_CACHE_HOME/miopen"
export HF_HOME="$XDG_CACHE_HOME/hf" HF_HUB_CACHE="$XDG_CACHE_HOME/hf/hub"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export IFAB_SDK_LOG_ROOT="$SS5_OPT_WORK/logs" SS5_ROUND1_CACHE_SCOPE="$XDG_CACHE_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL="$SS5_OPT_WORK/gitconfig"
[ -e "$GIT_CONFIG_GLOBAL" ] || : > "$GIT_CONFIG_GLOBAL"
export SS5_BASE_PYTHON="$ss5_afs/venvs/minimax-h3/bin/python"
[ -x "$SS5_BASE_PYTHON" ] || { echo "Missing read-only base Python: $SS5_BASE_PYTHON" >&2; return 1; }
set +u
source /opt/dtk/env.sh
set -u
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/usr/local/lib/python3.11/site-packages/torch.libs:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SS5_OPT_WORK/src/InfiniCore/python:$SS5_OPT_WORK/src/InfiniLM/python"
export FA_SO_PATH
FA_SO_PATH=$("$SS5_BASE_PYTHON" -B - <<'PY'
from pathlib import Path
import sys
found = [p for s in sys.path if s and ('site-packages' in s or 'dist-packages' in s)
         for p in Path(s).glob('flash_attn_2_cuda*.so')]
if not found:
    raise SystemExit('The unchanged vendor flash_attn_2_cuda extension is required')
print(found[0].resolve())
PY
)
export SS5_NT_ATTN="${SS5_NT_ATTN:-1}" SS5_NT_ADALN="${SS5_NT_ADALN:-1}"
case "$SS5_NT_ATTN$SS5_NT_ADALN" in 00|01|10|11) ;; *) echo 'Switches must be 00, 01, 10 or 11' >&2; return 1 ;; esac
