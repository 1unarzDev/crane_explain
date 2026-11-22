#!/usr/bin/env bash
# One command per received binned CSV session (WS-KIT2): arrival check -> adapter -> feature/target caches.
#   capxfer/field/process_session.sh SESSION_DIR [OUT_ROOT] [--seed-file SEED] [--bundle BUNDLE] [--force]
# Output: OUT_ROOT/<session>/ (default ~/capxfer_derived); never writes inside SESSION_DIR. Exit 2 on an arrival FAIL.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ $# -ge 1 ] || { sed -n 2,4p "$0"; exit 1; }
session="$1"; shift
out="${HOME}/capxfer_derived"
if [ $# -ge 1 ] && [[ "$1" != --* ]]; then out="$1"; shift; fi
py="${here}/.venv/bin/python"; [ -x "$py" ] || py="python3"
cd "$here"
PYTHONPATH="$here" nice -n 5 "$py" -m capxfer.field.process_session "$session" --out "$out" "$@"
