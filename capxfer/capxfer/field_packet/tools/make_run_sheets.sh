#!/usr/bin/env bash
# Regenerate every run sheet from a sealed, team-drawn seed (wraps capxfer/field/schedule.py; see make_run_sheets.py -h).
#   tools/make_run_sheets.sh main  OUT_DIR --seed-file SEALED/seed_main.txt --ids M1=..,M2=..,M3=..,M4=.. [--prereg-sha256 H] ...
#   tools/make_run_sheets.sh extra KIND OUT_DIR --seed-file SEALED/seed_KIND.txt [--module Mk] [--day 1|2] ...
# KIND: shakedown hooktest pilot sysid simval apriltag natural floor2 docking.
# Uses system python3 (numpy + PyYAML); no ROS, no torch. Run it on the robot laptop, from any directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${CAPXFER_REPO:-$(cd "$HERE/../../.." && pwd)}"
PY="${PYTHON:-python3}"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" "$HERE/make_run_sheets.py" "$@"
