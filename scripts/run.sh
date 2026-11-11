#!/bin/bash
set -euo pipefail
here=$(cd -P -- "$(dirname -- "$0")" && pwd)
source "$here/env.sh"
exec "$SS5_BASE_PYTHON" -B -u "$here/run.py" "$@"
