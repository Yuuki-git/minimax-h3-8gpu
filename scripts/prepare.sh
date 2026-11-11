#!/bin/bash
set -euo pipefail
here=$(cd -P -- "$(dirname -- "$0")" && pwd)
source "$here/env.sh"
"${SS5_VENDOR_PYTHON:-python3}" -B "$here/prepare.py"
cd -- "$SS5_OPT_WORK/src/xmake"
if [ ! -x "$SS5_OPT_WORK/deps/tools/bin/xmake" ]; then
  ./configure --prefix="$SS5_OPT_WORK/deps/tools"
  make -j 8
  make install
fi
echo "PREPARED=$SS5_OPT_WORK"
