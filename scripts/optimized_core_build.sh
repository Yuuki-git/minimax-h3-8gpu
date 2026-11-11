#!/bin/bash
# Run from the terminal or batch job's own workspace. AFS supplies inputs only.
set -euo pipefail

fail() { echo "SS5_V3D_FAILED: $*" >&2; exit 1; }
mode=${1:-check}
case "$mode" in inputs|configure|core-mha|lm) ;; *) fail '用法：bash ss5_workspace_v3d.sh inputs|configure|core-mha|lm' ;; esac

workspace=$(pwd -P)
parent=$(cd .. && pwd -P)
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
afs=$(cd -- "${IFAB_AFS_HOME:?需要 InfiniFabric 的 IFAB_AFS_HOME}" && pwd -P) || fail '个人存储不可读'
work="${SS5_OPT_WORK:?}"

check_inputs() (
  cd -- "$afs"
  local file resolved offline
  offline=$(realpath -m -- ss4-prep/offline)
  for file in \
              src-tars/xmake-repo-bd29d77f-src.tgz xmake-pkgs/boost-1.92.0.tar.gz \
              xmake-pkgs/pybind11-v3.1.0.zip tools/bin/xmake venvs/minimax-h3/bin/python; do
    [ -r "$file" ] && [ -s "$file" ] || fail "缺少输入：$afs/$file"
    resolved=$(realpath -e -- "$file")
    case "$resolved/" in "$offline/"*) fail "$file 仍引用 ss4-prep/offline" ;; esac
    case "$file" in *.tgz) gzip -t -- "$file" || fail "源码包损坏：$file" ;; esac
    echo "INPUT_OK=$file"
  done
  sha256sum -c - <<'SHA' || fail 'Boost 或 pybind11 的 SHA256 不符'
f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984  xmake-pkgs/boost-1.92.0.tar.gz
affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536  xmake-pkgs/pybind11-v3.1.0.zip
SHA
  echo 'SS5_OFFLINE_INPUTS_OK'
)

if [ "$mode" = inputs ]; then
  check_inputs
  exit 0
fi

[ "$workspace" != / ] || fail '当前目录为 /，请先进入 IFAB_WORKSPACE_ROOT'
case "$workspace/" in "$afs/"*) fail '当前目录在个人 AFS 内。请在登录终端或有卡会话打开时的初始工作目录执行' ;; esac
user_dir=$(cd -- "${HOME:?需要会话 HOME}" && pwd -P) || fail '无法解析会话 HOME'
case "$user_dir/" in "$parent/"*) ;; *) fail '会话 HOME 不在当前目录或上一级内，无法限制子工具的默认写入位置' ;; esac
unset PYTHONOPTIMIZE
if [ "$mode" = core-flash ] || [ "$mode" = core-mha ]; then
  platform_workspace=$(cd -- "${IFAB_WORKSPACE_ROOT:?缺少本次临时工作区}" && pwd -P)
  [ "$workspace" = "$platform_workspace" ] || fail '请先进入 IFAB_WORKSPACE_ROOT'
fi

python3 -B - "$workspace" "$afs" <<'PY' || fail '工作目录预检失败'
import errno
from pathlib import Path
import shutil
import sys
import tempfile

workspace, afs = map(Path, sys.argv[1:])
free = shutil.disk_usage(workspace).free
print(f"SS5_WORKSPACE={workspace}")
print(f"SS5_WORKSPACE_FREE_BYTES={free}")
if workspace.stat().st_dev == afs.stat().st_dev:
    raise SystemExit("当前目录与个人 AFS 在同一设备，无法确认独立构建空间")
if free < 4 * 1024**3:
    raise SystemExit("当前目录可用空间少于 4 GiB（本脚本的启动下限）")
with tempfile.TemporaryDirectory(prefix="ss5-fs-", dir=workspace) as path:
    probe = Path(path)
    scratch = probe / "source.tmp"
    (scratch / "pkg").mkdir(parents=True)
    try:
        (scratch / "pkg").rename(probe / "nested")
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        print("SS5_RENAME_NESTED=EXDEV")
    else:
        print("SS5_RENAME_NESTED=OK")
    (scratch / "file").write_text("probe\n", encoding="utf-8")
    scratch.rename(probe / "source")
    assert (probe / "source/file").read_text(encoding="utf-8") == "probe\n"
    print("SS5_RENAME_SAME_PARENT=OK")
PY

lib="$here/runtime_env.sh"
[ -r "$lib" ] || fail '缺少之前使用的 runtime_env.sh'
[ -r "$here/ss5_xmake_extract_v3.py" ] || fail '同目录缺少 ss5_xmake_extract_v3.py'


# Explicit state directories keep all outputs inside the starting workspace.
for name in '' /tmp /src /sources /logs /infini /tools /build-core /build-core-mha /build-lm /cache \
            /xmake-global /xmake-cache /xmake-packages /xmake-core /xmake-core-mha /xmake-lm; do
  dir="$work$name"
  [ "$(realpath -m -- "$dir")" = "$dir" ] || fail "工作路径包含符号链接：$dir"
  mkdir -p -- "$dir"
done
[ ! -L "$work/run.lock" ] || fail 'run.lock 是符号链接'
exec 9>>"$work/run.lock"
flock -n 9 || fail '已有本脚本在当前工作区执行'

export TMPDIR="$work/tmp" TMP="$work/tmp" TEMP="$work/tmp" SS4_STAGE="$work/src"
export XMAKE_TMPDIR="$TMPDIR" XMAKE_RAMDIR="$TMPDIR" XMAKE_GLOBALDIR="$work/xmake-global"
export XMAKE_PKG_CACHEDIR="$work/xmake-cache" XMAKE_PKG_INSTALLDIR="$work/xmake-packages"
export XMAKE_STATS=false
export XDG_CACHE_HOME="$work/cache" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export INFINI_ROOT="$work/infini" XMAKE_ROOT=y CMAKE_BUILD_PARALLEL_LEVEL=8
export IFAB_SDK_LOG_ROOT="$work/logs"
source "$lib"
export PYTHON="$VENV_PY"
export SS5_STAGE="$STAGE"
readonly SS5_WORKSPACE="$workspace"
export SS5_WORKSPACE
[ -x "$VENV_PY" ] && [ -x "$XMAKE" ] || fail '找不到现有 venv 或 xmake'

set +u
source /opt/dtk/env.sh || fail 'DTK 环境加载失败'
set -u
export PATH="/opt/dtk/cuda/cuda/bin:$PATH"
runtime_env

project_options=()
if [ "$mode" = core-mha ]; then
  mha_project="$STAGE/InfiniCore/.ss5-core-mha.lua"
  project_options=(--file="$mha_project")
fi

LOGDIR=$(mktemp -d -- "$work/logs/$mode.XXXXXXXX")
echo "LOG_DIR=$LOGDIR"
run_logged() {
  local name=$1; shift
  "$@" 2>&1 | tee "$LOGDIR/$name.log" || fail "$name，详见 $LOGDIR/$name.log"
}

if [ "$mode" != reference ]; then
  export SS5_SYSTEM_TAR=/usr/bin/tar SS5_SYSTEM_UNZIP=/usr/bin/unzip
  # Use a fresh directory so an interrupted install cannot affect this attempt.
  tool_dir=$(mktemp -d -- "$work/tools/extract.XXXXXXXX") || fail '无法建立本轮解包工具目录'
  for tool in tar unzip; do
    [ -x "/usr/bin/$tool" ] || fail "缺少系统 $tool"
    wrapper="$tool_dir/$tool"
    cat -- "$here/ss5_xmake_extract_v3.py" > "$wrapper" || fail "无法写入 $wrapper"
    chmod 0755 -- "$wrapper" || fail "无法设置 $wrapper 的执行权限"
  done
  export PATH="$tool_dir:$PATH"
  run_logged tar-tool "$tool_dir/tar" --version
  run_logged unzip-tool "$tool_dir/unzip" -v
  echo "SS5_XMAKE_TOOLS_OK=$tool_dir"
  run_logged xmake-version "$XMAKE" --version
  grep -Fq -- '3.0.5+HEAD.f20295f' "$LOGDIR/xmake-version.log" || fail 'xmake 版本与已核对的版本不符'
  for archive in boost-1.92.0.tar.gz pybind11-v3.1.0.zip; do
    [ -r "$XMAKE_PKGS/$archive" ] || fail "缺少 03 v2 使用的离线包别名：$XMAKE_PKGS/$archive"
  done
fi

# Point the existing restore helper only at the pinned inputs prepared in 01.
TARS="$work/sources"
restore() {
  local repo=$1 commit=$2 archive actual
  shift 2
  if [ "$repo" = InfiniCore ] || [ "$repo" = InfiniLM ]; then
    local expected
    expected=$("$VENV_PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["sources"][sys.argv[2]]["commit"])' "$here/../source-lock.json" "$repo")
    [ -d "$STAGE/$repo/.git" ] || fail "Run fetch_sources.sh first: $repo"
    actual=$(git -C "$STAGE/$repo" rev-parse HEAD)
    [ "$actual" = "$expected" ] || fail "Source commit differs: $repo $actual"
    echo "SOURCE_COMMIT=$repo $actual"
    return
  fi
  archive="$repo-${commit:0:8}-src.tgz"
  [ -r "$afs/src-tars/$archive" ] || fail "缺少 01 的源码输入 $archive"
  if [ ! -L "$TARS/$archive" ]; then
    ln -s -- "$afs/src-tars/$archive" "$TARS/$archive"
  fi
  [ "$(readlink -- "$TARS/$archive")" = "$afs/src-tars/$archive" ] || fail "源码输入路径不符：$archive"
  [ "$(realpath -m -- "$STAGE/$repo")" = "$STAGE/$repo" ] || fail "源码工作目录包含符号链接：$repo"
  run_logged "restore-$repo" ensure_stage "$repo" "$@"
  actual=$(git -c "safe.directory=$STAGE/$repo" -C "$STAGE/$repo" rev-parse HEAD)
  [ "$actual" = "$commit" ] || fail "$repo 源码提交不符：$actual"
  echo "SOURCE_COMMIT=$repo $actual"
}

setup_xmake() {
  restore xmake-repo bd29d77f242e247589457638272d9936a24a3fe0 packages/b/boost/xmake.lua packages/p/pybind11/xmake.lua
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

probe_hygon_mha_dispatch() {
  local phase=$1 library="$LOGDIR/mha-dispatch.so"
  local core="$INFINI_ROOT/lib/libinfinicore_cpp_api.so"
  if [ ! -s "$library" ]; then
    run_logged build-mha-dispatch g++ -std=c++17 -Wall -Wextra -Werror -O2 -shared -fPIC \
      -I"$STAGE/InfiniCore/include" -x c++ - -x none -L"$INFINI_ROOT/lib" \
      -Wl,-z,defs -linfinicore_cpp_api -o "$library" <<'CPP'
#include <infinicore/ops/mha.hpp>

extern "C" int ss5_hygon_mha_dispatch() {
    using infinicore::Device;
    using infinicore::op::MultiheadAttention;
    const auto device = Device::Type::HYGON;
    int plan = MultiheadAttention::plan_dispatcher().lookup(device) != nullptr;
    int run = MultiheadAttention::run_dispatcher().lookup(device) != nullptr;
    int cleanup = MultiheadAttention::cleanup_dispatcher().lookup(device) != nullptr;
    return plan | (run << 1) | (cleanup << 2);
}
CPP
  fi
  run_logged "mha-dispatch-$phase" "$VENV_PY" -B - "$core" "$library" "$phase" <<'PY'
import ctypes
from pathlib import Path
import sys

import torch

core, library, phase = sys.argv[1:]
ctypes.CDLL(core, mode=ctypes.RTLD_GLOBAL)
probe = ctypes.CDLL(library)
probe.ss5_hygon_mha_dispatch.argtypes = []
probe.ss5_hygon_mha_dispatch.restype = ctypes.c_int
bits = probe.ss5_hygon_mha_dispatch()
loaded = {line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
          if line.endswith("/libinfinicore_cpp_api.so")}
if loaded != {str(Path(core).resolve())}:
    raise SystemExit(f"Core 动态库来源不符 {sorted(loaded)}")
print(f"SS5_HYGON_MHA_DISPATCH phase={phase} plan={bits & 1} run={(bits >> 1) & 1} cleanup={(bits >> 2) & 1}")
if phase == "after" and bits != 7:
    raise SystemExit("海光 mha 的分发表项不完整")
PY
}

probe_hygon_mha_tensor() {
  PYTHONPATH="$STAGE/InfiniCore/python" PYTHONFAULTHANDLER=1 \
    run_logged mha-tensor "$VENV_PY" -B - <<'PY'
import torch

import infinicore

device = "cuda:0"
torch.cuda.set_device(device)
arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
if torch.cuda.device_count() != 8 or arch != "gfx936":
    raise SystemExit(f"需要 8 卡 gfx936 会话，当前卡数 {torch.cuda.device_count()}，卡 0 {arch}")
infinicore.set_device(infinicore.device(device))
shape = (1, 4, 2, 128)
q = torch.zeros(shape, dtype=torch.bfloat16, device=device)
v = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1).expand(shape).to(device, q.dtype).contiguous()
out = torch.empty_like(q)
torch.cuda.synchronize()
native_q, native_v, native_out = map(infinicore.from_torch, (q, v, out))
print(f"SS5_HYGON_MHA_BEGIN device={device} shape={shape} dtype={q.dtype} scale={128 ** -0.5}", flush=True)
infinicore.mha(native_q, native_q, native_v, scale=128 ** -0.5, out=native_out)
infinicore.sync_stream()
torch.testing.assert_close(out.cpu(), torch.full(shape, 1.5, dtype=q.dtype), atol=0, rtol=0)
print("SS5_HYGON_MHA_OK")
PY
}

prepare_core_mha_project() {
  [ "$(realpath -m -- "$mha_project")" = "$mha_project" ] \
    || fail '海光 mha 构建配置路径包含符号链接'
  run_logged verify-core-mha sha256sum --check --strict - <<'SHA'
dca38aa124325cae265efe72cd99c817f756d236530e3a5125c7c980ba83915a  xmake.lua
SHA
  run_logged prepare-core-mha "$VENV_PY" -B - "$mha_project" <<'PY'
import hashlib
from pathlib import Path
import sys

base = Path("xmake.lua").read_text(encoding="utf-8")
anchor = '    else\n        remove_files("src/infinicore/ops/*/hygon/*.cc")\n    end\n'
if base.count(anchor) != 1:
    raise SystemExit("固定版 Core 的海光源码选择规则不符")
selection = '''    elseif has_config("hygon-dcu") and has_config("aten") then
        -- The upstream dense Hygon MHA has no external FlashAttention calls.
        remove_files("src/infinicore/ops/mha_kvcache/hygon/*.cc")
        remove_files("src/infinicore/ops/multi_head_attention_varlen/hygon/*.cc")
        add_files("src/infinicore/ops/multi_head_attention/hygon/mha_flashattn_hygon.cc",
                  {defines = "ENABLE_FLASH_ATTN"})
        add_includedirs("src/infinicore/ops/ss5_ninetoothed/generated")
        add_links("ss5_s4_attention", "ss5_s3_adaln", "ss5_s3_gate")
'''
project = Path(sys.argv[1])
content = base.replace(anchor, selection + anchor).encode("utf-8")
project.write_bytes(content)
print(f"SS5_CORE_MHA_PROJECT={project}")
print(f"SS5_CORE_MHA_PROJECT_SHA256={hashlib.sha256(content).hexdigest()}")
PY
  diff -u -- xmake.lua "$mha_project" > "$LOGDIR/core-mha-build.diff" || [ "$?" -eq 1 ] \
    || fail '无法记录海光 mha 构建配置差异'
}

configure_core() {
  restore InfiniCore f32e2fbb2a9b2ca912b5711f4f3836c403196a74 xmake.lua setup.py python/infinicore/__init__.py \
    third_party/spdlog/include/spdlog/spdlog.h submodules/InfiniRT/CMakeLists.txt \
    submodules/InfiniOps/CMakeLists.txt submodules/InfiniCCL/CMakeLists.txt
  cd -- "$STAGE/InfiniCore"
  local component=core
  if [ "$mode" = core-mha ]; then
    component=core-mha
    prepare_core_mha_project
  fi
  export XMAKE_CONFIGDIR="$work/xmake-$component"
  local flash_options=()
  if [ "$mode" = core-flash ]; then
    export PATH="$VENV/bin:$PATH"
    flash_options+=(--flash-attn="$flash_root")
  fi
  if [[ "$mode" = core-flash || "$mode" = core-mha ]] && [ -s "$INFINI_ROOT/lib/libinfinicore_cpp_api.so" ]; then
    probe_hygon_mha_dispatch before
  fi
  setup_xmake
  configure_with_dependencies "$component" \
    --hygon-dcu=y --hygon-arch=gfx936 --cuda=/opt/dtk/cuda/cuda --ccl=y --aten=y "${flash_options[@]}"
  echo 'SS5_STAGE03_CONFIG_OK'
}

check_core() {
  local lib
  for lib in infinirt infiniop infiniccl infinicore_cpp_api; do
    [ -s "$INFINI_ROOT/lib/lib$lib.so" ] || fail "缺少 lib$lib.so，请在本会话完成 core"
  done
}

check_extension() {
  local repo=$1 package=$2 extension=$3 path
  path=$(find "$STAGE/$repo/python/$package" -type f -name "$extension*.so" -size +0c -print -quit)
  [ -n "$path" ] || fail "未找到 $extension 动态库"
  sha256sum -- "$path" | tee "$LOGDIR/$extension.sha256"
}

build_core() {
  configure_core
  local target
  run_logged build-core "$XMAKE" build "${project_options[@]}" -j 8 infinicore_cpp_api
  for target in infinirt infiniop infiniccl infinicore_cpp_api; do
    run_logged "install-$target" "$XMAKE" install "${project_options[@]}" "$target"
  done
  check_core
  run_logged build-_infinicore "$XMAKE" build "${project_options[@]}" -j 8 _infinicore
  run_logged install-_infinicore "$XMAKE" install "${project_options[@]}" _infinicore
  check_extension InfiniCore infinicore _infinicore
  if [ "$mode" = core-mha ]; then
    local library="$INFINI_ROOT/lib/libinfinicore_cpp_api.so"
    run_logged core-mha-needed readelf -d -- "$library"
    if grep -E 'NEEDED.*(libflash_attention\.so|flash_attn_2_cuda)' "$LOGDIR/core-mha-needed.log"; then
      fail '仅启用稠密 mha 的 Core 仍链接到厂商 FlashAttention 库'
    fi
    run_logged core-mha-sha256 sha256sum -- "$library"
  fi
  if [ "$mode" = core-flash ] || [ "$mode" = core-mha ]; then
    probe_hygon_mha_dispatch after
    probe_hygon_mha_tensor
  fi
  echo 'STAGE03_OK'
}

build_lm() {
  check_core
  restore InfiniLM 65deb91f840ebaef752d028fad2f3217ce0950f4 xmake.lua setup.py examples/minimax_h3.py \
    third_party/spdlog/include/spdlog/spdlog.h third_party/json/single_include/nlohmann/json.hpp
  cd -- "$STAGE/InfiniLM"
  export XMAKE_CONFIGDIR="$work/xmake-lm"
  setup_xmake
  local abi
  abi=$("$VENV_PY" -B -c 'import torch; print(int(torch.compiled_with_cxx11_abi()))') || fail '无法读取 Torch C++ ABI'
  case "$abi" in 0|1) ;; *) fail "Torch C++ ABI 输出无效：$abi" ;; esac
  export INFINILM_CXX11_ABI="$abi"
  configure_with_dependencies lm --cxx11-abi="$abi"
  run_logged build-_infinilm "$XMAKE" build -j 8 _infinilm
  run_logged install-_infinilm "$XMAKE" install _infinilm
  check_extension InfiniLM infinilm _infinilm
  echo 'STAGE04_OK'
}

case "$mode" in
  configure) configure_core ;;
  core-mha) build_core ;;
  lm) build_lm ;;
esac
