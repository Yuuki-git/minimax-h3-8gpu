#!/bin/bash
set -euo pipefail
here=$(cd -P -- "$(dirname -- "$0")" && pwd)
source "$here/optimized_env.sh"
exec "$SS5_BASE_PYTHON" -B "$here/fetch_sources.py"
