#!/usr/bin/env bash
# Continue the existing CTC checkpoint, leaving five minutes for a clean save.
set -euo pipefail
cd /work/gm64/m64000/OTTC
CIF_PY=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python
CIF_NOW=$(date +%s)
# This interactive allocation began at 2026-10-16 15:15:40 JST, for one hour.
CIF_DEADLINE=$(date -d '2026-10-16 16:15:40 +0900' +%s)
CIF_REMAIN=$((CIF_DEADLINE-CIF_NOW-300))
if (( CIF_REMAIN < 60 )); then exit 0; fi
export CIF_CTC_WEIGHT=0.5 CIF_OTTC_WEIGHT=0
export CIF_MAX_WALL_SECONDS="$CIF_REMAIN"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1 \
  > "reproduction/cif_timit_20261016/train40_ctc_interactive_${PBS_JOBID}.telemetry.csv" &
CIF_TELEMETRY=$!
trap 'kill "$CIF_TELEMETRY" 2>/dev/null || true' EXIT
bash reproduction/cif_timit_20261016/interactive.sh --epochs 40
