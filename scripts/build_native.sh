#!/bin/bash
set -euo pipefail
here=$(cd -P -- "$(dirname -- "$0")" && pwd)
source "$here/env.sh"
work="$SS5_OPT_WORK"
STAGE="$work/src"
VENV_PY="$SS5_BASE_PYTHON"
XMAKE="$work/deps/tools/bin/xmake"
XMAKE_PKGS="$work/deps/xmake-pkgs"
export PYTHON="$VENV_PY"
export SS5_WORKSPACE="$ss5_workspace"
export SS5_SYSTEM_TAR=/usr/bin/tar SS5_SYSTEM_UNZIP=/usr/bin/unzip
mkdir -p -- "$work/tools" "$work/logs/native"
for tool in tar unzip; do
  cp -- "$here/xmake_extract.py" "$work/tools/$tool"
  chmod 0755 -- "$work/tools/$tool"
done
export PATH="$work/tools:$PATH"
LOGDIR="$work/logs/native"
fail() { echo "$*" >&2; exit 1; }
run_logged() { local name=$1; shift; "$@" 2>&1 | tee "$LOGDIR/$name.log"; }
project_options=()
setup_xmake() {
  run_logged xmake-global "$XMAKE" global --network=private --pkg_searchdirs="$XMAKE_PKGS"
  "$XMAKE" repo "${project_options[@]}" --remove ss5-official >/dev/null 2>&1 || true
  run_logged xmake-repo "$XMAKE" repo "${project_options[@]}" --add ss5-official "$STAGE/xmake-repo"
}
configure_with_dependencies() {
  local component=$1; shift
  run_logged "configure-$component" "$XMAKE" f "${project_options[@]}" -y -cv --require=no -m release \
    --builddir="$work/build-$component" "$@"
  run_logged "packages-$component" "$XMAKE" require "${project_options[@]}" -y -v -j 8
}
cd -- "$STAGE/InfiniCore"
mha_project="$STAGE/InfiniCore/.ss5-core-mha.lua"
project_options=(--file="$mha_project")
"$VENV_PY" -B - "$mha_project" <<'PY'
from pathlib import Path
import sys
base = Path('xmake.lua').read_text()
anchor = '    else\n        remove_files("src/infinicore/ops/*/hygon/*.cc")\n    end\n'
if base.count(anchor) != 1:
    raise ValueError('The pinned Core build definition changed')
selection = '''    elseif has_config("hygon-dcu") and has_config("aten") then
        remove_files("src/infinicore/ops/mha_kvcache/hygon/*.cc")
        remove_files("src/infinicore/ops/multi_head_attention_varlen/hygon/*.cc")
        add_files("src/infinicore/ops/multi_head_attention/hygon/mha_flashattn_hygon.cc",
                  {defines = "ENABLE_FLASH_ATTN"})
        add_includedirs("src/infinicore/ops/ss5_ninetoothed/generated")
        add_links("ss5_s4_attention", "ss5_s3_adaln", "ss5_s3_gate")
'''
Path(sys.argv[1]).write_text(base.replace(anchor, selection + anchor))
PY
export XMAKE_CONFIGDIR="$work/xmake-core"
setup_xmake
configure_with_dependencies core --hygon-dcu=y --hygon-arch=gfx936 --cuda=/opt/dtk/cuda/cuda --ccl=y --aten=y
run_logged build-core "$XMAKE" build "${project_options[@]}" -j 8 infinicore_cpp_api
for target in infinirt infiniop infiniccl infinicore_cpp_api; do
  run_logged "install-$target" "$XMAKE" install "${project_options[@]}" "$target"
done
run_logged build-core-python "$XMAKE" build "${project_options[@]}" -j 8 _infinicore
run_logged install-core-python "$XMAKE" install "${project_options[@]}" _infinicore
cd -- "$STAGE/InfiniLM"
project_options=()
export XMAKE_CONFIGDIR="$work/xmake-lm"
setup_xmake
abi=$("$VENV_PY" -B -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))')
case "$abi" in 0|1) ;; *) fail 'Invalid Torch C++ ABI' ;; esac
export INFINILM_CXX11_ABI="$abi"
configure_with_dependencies lm --cxx11-abi="$abi"
run_logged build-lm "$XMAKE" build -j 8 _infinilm
run_logged install-lm "$XMAKE" install _infinilm
