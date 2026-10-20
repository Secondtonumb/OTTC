#!/usr/bin/env bash
set -euo pipefail
source /work/gm64/m64000/OTTC/reproduction/cif_quantity_20261016/environment.sh
EVAL_ROOT=/work/gm64/m64000/OTTC/reproduction/cif_libri100_eval_20261017
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 2 > "$EVAL_ROOT/telemetry_${PBS_JOBID}.csv" &
EVAL_TELEMETRY=$!
trap 'kill "$EVAL_TELEMETRY" 2>/dev/null || true' EXIT
"$CIF_PY" -u -m reproduction.cif_libri100_eval_20261017.evaluate_saved 2>&1 | tee "$EVAL_ROOT/eval_${PBS_JOBID}.log"
