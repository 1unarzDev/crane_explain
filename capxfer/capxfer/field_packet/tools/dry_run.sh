#!/usr/bin/env bash
# Dry run of the packet tools on SYNTHETIC SIMULATION sessions (needs the capxfer venv: torch for the simulator/loader).
#   capxfer/field_packet/tools/dry_run.sh [--work DIR] [--trials N]      (~2.5 min on CPU; writes only under --work)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${CAPXFER_REPO:-$(cd "$HERE/../../.." && pwd)}"
PY="${PYTHON:-$REPO/.venv/bin/python}"
cd "$REPO"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" exec "$PY" "$HERE/dry_run.py" "$@"
