#!/bin/bash
# Shared paths and upstream-source extraction for the eight-card build.
AFS="${IFAB_AFS_HOME:?需要 IFAB_AFS_HOME}"
TMP="${TMPDIR:?需要会话内 TMPDIR}"
STAGE="${SS4_STAGE:?需要当前会话源码目录}"
TARS="$AFS/src-tars"
VENV="$AFS/venvs/minimax-h3"
VENV_PY="$VENV/bin/python"
XMAKE="$AFS/tools/bin/xmake"
XMAKE_PKGS="$AFS/xmake-pkgs"
export PATH="$AFS/tools/bin:$PATH"
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL="$AFS/.gitconfig"

runtime_env() {
  local root="${INFINI_ROOT:-$AFS/infini}" tp=/usr/local/lib/python3.11/site-packages
  export LD_LIBRARY_PATH="$root/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:$tp/torch/lib:$tp/torch.libs:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"
}

extract_tar() {
  local archive="$1" dest="$2" err="$TMP/extract-tar-$$.err" line path bad=0
  if tar -xzf "$archive" -C "$dest" 2> "$err"; then
    rm -f "$err"
    return 0
  fi
  while IFS= read -r line; do
    case "$line" in
      "tar: Exiting with failure status due to previous errors") ;;
      "tar: "*": Cannot change mode to "*)
        path="${line#tar: }"
        path="${path%%: Cannot change mode to *}"
        if [ ! -L "$dest/$path" ]; then echo "$line"; bad=1; fi
        ;;
      *) echo "$line"; bad=1 ;;
    esac
  done < "$err"
  rm -f "$err"
  return "$bad"
}

ensure_stage() {
  local name="$1"; shift
  local dir="$STAGE/$name" tar_file f ok
  if [ "$#" -lt 1 ]; then echo "[stage] ensure_stage 需要至少一个关键文件参数"; return 1; fi
  if [ -d "$dir" ]; then
    ok=0
    for f in "$@"; do [ -e "$dir/$f" ] || { ok=1; break; }; done
    [ "$ok" = 0 ] && { echo "[stage] $name 已就绪: $dir"; return 0; }
  fi
  mkdir -p "$STAGE"
  local matches=("$TARS/$name"-*-src.tgz)
  [ "${#matches[@]}" -eq 1 ] || { echo "需要唯一的固定源码输入 $name"; return 1; }
  tar_file="${matches[0]}"
  if [ ! -f "$tar_file" ]; then
    echo "[stage] 缺少 $TARS/$name-*.tgz —— 请先运行 scripts/reproduce.sh --prepare-only"
    return 1
  fi
  echo "[stage] 恢复到 $dir  (来源: ${tar_file##*/})"
  [ ! -e "$dir" ] || { echo "源码目录不完整，请另开会话构建：$dir"; return 1; }
  extract_tar "$tar_file" "$STAGE" || return 1
  for f in "$@"; do
    [ -e "$dir/$f" ] || { echo "[stage] 恢复后仍缺 $name/$f"; return 1; }
  done
  echo "[stage] $name 就绪"
  return 0
}
