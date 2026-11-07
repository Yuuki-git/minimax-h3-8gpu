#!/bin/bash
# Run on an online Linux machine/WSL, outside personal AFS. Outputs stay in dist.
set -euo pipefail
repo_root=$(cd -P -- "$(dirname -- "$0")/.." && pwd)
case "$(uname -s)" in Linux) ;; *) echo 'Use a Linux filesystem or WSL to preserve upstream symlinks' >&2; exit 1 ;; esac
prep_root="$repo_root/dist/source-prep"
input_root="$repo_root/dist/platform-inputs"
mkdir -p -- "$prep_root" "$input_root/src-tars"

prepare() {
  local name=$1 url=$2 commit=$3
  shift 3
  local source="$prep_root/$name" output="$input_root/src-tars/$name-${commit:0:8}-src.tgz"
  if [ -e "$source" ]; then
    [ -d "$source/.git" ] || { echo "Unexpected source directory $source" >&2; exit 1; }
    [ "$(git -C "$source" rev-parse HEAD)" = "$commit" ] || { echo "Commit mismatch $source" >&2; exit 1; }
    [ -z "$(git -C "$source" status --porcelain --untracked-files=no --ignore-submodules=dirty)" ] || { echo "Tracked source changed $source" >&2; exit 1; }
  else
    git init -q "$source"
    git -C "$source" remote add origin "$url"
    git -C "$source" fetch --depth 1 origin "$commit"
    git -c core.autocrlf=false -C "$source" checkout --detach FETCH_HEAD
  fi
  if [ "$#" -gt 0 ]; then
    git -c core.autocrlf=false -C "$source" submodule update --init --recursive --depth 1 -- "$@"
  fi
  tar -czf "$output" -C "$prep_root" "$name"
  sha256sum "$output"
}

prepare InfiniCore https://github.com/InfiniTensor/InfiniCore.git f32e2fbb2a9b2ca912b5711f4f3836c403196a74 \
  third_party/spdlog submodules/InfiniRT submodules/InfiniOps submodules/InfiniCCL
prepare InfiniLM https://github.com/InfiniTensor/InfiniLM.git 65deb91f840ebaef752d028fad2f3217ce0950f4 \
  third_party/spdlog third_party/json
prepare xmake-repo https://github.com/xmake-io/xmake-repo.git bd29d77f242e247589457638272d9936a24a3fe0
echo "SOURCE_INPUTS=$input_root/src-tars"
echo 'Also provide xmake 3.0.5+HEAD.f20295f at $IFAB_AFS_HOME/tools/bin/xmake.'
