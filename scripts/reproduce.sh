#!/bin/bash
# Prepare fixed inputs, build upstream libraries, then run the official FL2VA pair.
set -euo pipefail
repo_root=$(cd -P -- "$(dirname -- "$0")/.." && pwd)
fail() { echo "REPRODUCE_FAILED: $*" >&2; exit 1; }
prepare_only=false
build_only=false
skip_build=false
runtime=''
run_args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --prepare-only) prepare_only=true; shift ;;
    --build-only) build_only=true; shift ;;
    --offline) shift ;; # Target execution only consumes prepared local inputs.
    --skip-build) skip_build=true; shift ;;
    --runtime)
      [ "$#" -ge 2 ] || fail '--runtime 缺少路径'
      runtime=$(realpath -e -- "$2") || fail '运行包目录不存在'
      shift 2 ;;
    --first-frame|--model-path|--reference-video|--output-dir)
      [ "$#" -ge 2 ] || fail "$1 缺少参数"
      case "$1" in
        --first-frame|--reference-video) [ -r "$2" ] && [ -s "$2" ] || fail "输入文件不可读：$2" ;;
        --model-path) [ -d "$2" ] || fail "模型目录不存在：$2" ;;
        --output-dir) [ ! -e "$2" ] || fail "输出目录已存在：$2" ;;
      esac
      run_args+=("$1" "$(realpath -m -- "$2")"); shift 2 ;;
    --help|-h)
      echo '准备：bash scripts/reproduce.sh --prepare-only'
      echo '编译并保存：bash scripts/reproduce.sh --offline --build-only'
      echo '运行：bash scripts/reproduce.sh --offline --first-frame /个人路径/first_frame.png'
      echo '复用：增加 --skip-build [--runtime /个人AFS/运行包目录] [--reference-video /参考视频.mp4]'
      exit 0 ;;
    *) fail "未知参数：$1" ;;
  esac
done
if "$prepare_only"; then
  [ -z "$runtime" ] && ! "$skip_build" && ! "$build_only" && [ "${#run_args[@]}" -eq 0 ] || fail '--prepare-only 单独使用'
  bash "$repo_root/scripts/prepare_sources.sh"
  python3 -B "$repo_root/scripts/build_support_bundles.py"
  mkdir -p -- "$repo_root/dist/platform-inputs/ss4-prep"
  cp -- "$repo_root/dist/diffusers-0.40.0-py3-none-any.whl" \
    "$repo_root/dist/ss5_compare_metrics.pyz" "$repo_root/dist/platform-inputs/ss4-prep/"
  echo "PREPARED_INPUTS=$repo_root/dist/platform-inputs"
  exit 0
fi
if "$build_only"; then
  ! "$skip_build" && [ -z "$runtime" ] && [ "${#run_args[@]}" -eq 0 ] \
    || fail '--build-only 与生成、恢复参数分开使用'
else
  [[ " ${run_args[*]} " == *' --first-frame '* ]] || fail '请用 --first-frame 指定主办方统一首帧'
fi
[ -z "$runtime" ] || "$skip_build" || fail '--runtime 需与 --skip-build 一起使用'
afs=$(cd -P -- "${IFAB_AFS_HOME:?缺少个人 AFS}" && pwd)
workspace=$(cd -P -- "${IFAB_WORKSPACE_ROOT:?请进入 8 卡会话}" && pwd)
work="$workspace/ss5-v3-work"
venv_python="$afs/venvs/minimax-h3/bin/python"
[ -x "$venv_python" ] || fail "缺少主办方基础环境所需独立 venv：$venv_python"

# Source preparation is done online; target execution never fetches dependencies.
if ! "$skip_build"; then
  for file in src-tars/InfiniCore-f32e2fbb-src.tgz src-tars/InfiniLM-65deb91f-src.tgz \
              src-tars/xmake-repo-bd29d77f-src.tgz xmake-pkgs/boost-1.92.0.tar.gz \
              xmake-pkgs/pybind11-v3.1.0.zip ss4-prep/diffusers-0.40.0-py3-none-any.whl \
              ss4-prep/ss5_compare_metrics.pyz; do
    if [ ! -s "$afs/$file" ]; then
      input="$repo_root/dist/platform-inputs/$file"
      [ -s "$input" ] || fail "缺少 $afs/$file；先在联网 Linux 准备并上传 dist/platform-inputs"
      mkdir -p -- "$(dirname -- "$afs/$file")"
      cp -- "$input" "$afs/$file"
    fi
  done
  (
    cd -P -- "$afs"
    "$venv_python" -B "$repo_root/scripts/ss5_stage02_diffusers.py" \
      --wheel "$afs/ss4-prep/diffusers-0.40.0-py3-none-any.whl"
    "$venv_python" -B "$afs/ss4-prep/ss5_compare_metrics.pyz"
  )
  (
    cd -P -- "$workspace"
    bash "$repo_root/scripts/ss5_workspace_v3d.sh" inputs
    bash "$repo_root/scripts/ss5_workspace_v3d.sh" core-mha
    bash "$repo_root/scripts/ss5_workspace_v3d.sh" lm
  )
  if "$build_only"; then
    (
      cd -P -- "$afs"
      python3 -B "$repo_root/scripts/ss5_native_checkpoint_v2.py" save
    )
    echo 'SS5_BUILD_ONLY_COMPLETE'
    exit 0
  fi
elif [ -n "$runtime" ]; then
  [ ! -e "$work" ] || fail '当前会话已有构建目录，请直接 --skip-build 或另开会话恢复'
  (
    cd -P -- "$workspace"
    python3 -B "$repo_root/scripts/ss5_native_checkpoint_v2.py" restore "$runtime"
  )
fi
[ -d "$work" ] && [ ! -L "$work" ] || fail '当前会话尚无编译产物，请构建或用 --runtime 恢复'
for directory in tmp cache logs; do mkdir -p -- "$work/$directory"; done
[ ! -L "$work/run.lock" ] || fail '运行锁是符号链接'
exec 9>>"$work/run.lock"
flock -n 9 || fail '当前工作区正在构建或推理'
export TMPDIR="$work/tmp" TMP="$work/tmp" TEMP="$work/tmp" SS4_STAGE="$work/src"
export INFINI_ROOT="$work/infini" XDG_CACHE_HOME="$work/cache"
source "$repo_root/scripts/runtime_env.sh"
set +u
source /opt/dtk/env.sh
set -u
runtime_env
export PYTHONPATH="$work/src/InfiniCore/python:$work/src/InfiniLM/python"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$XDG_CACHE_HOME/hf" HF_HUB_CACHE="$XDG_CACHE_HOME/hf/hub"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MIOPEN_USER_DB_PATH="$XDG_CACHE_HOME/miopen-db" MIOPEN_CUSTOM_CACHE_DIR="$XDG_CACHE_HOME/miopen-cache"
export IFAB_SDK_LOG_ROOT="$work/logs"
unset PYTHONOPTIMIZE
export FA_SO_PATH
FA_SO_PATH=$(find "$VENV"/lib/python*/site-packages -maxdepth 1 -name 'flash_attn_2_cuda*.so' -print -quit)
[ -s "$FA_SO_PATH" ] || fail '缺少 venv 中的厂商 FlashAttention 扩展'
cd -P -- "$afs"
exec "$venv_python" -B "$repo_root/scripts/run_fl2va.py" "${run_args[@]}"
