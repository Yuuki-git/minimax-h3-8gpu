#!/bin/bash
set -euo pipefail
here=$(cd -P -- "$(dirname -- "$0")" && pwd)
exec bash "$here/run_optimized.sh" compare "$@"
