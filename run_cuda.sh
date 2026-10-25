#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
task_python="${SEISMIC_JOINT_PYTHON:-python3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$task_python" -m seismic_joint "$@"
