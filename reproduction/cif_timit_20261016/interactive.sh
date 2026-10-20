#!/usr/bin/env bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
export PYTHONPATH="$PWD/reproduction/vendor:$PWD"
export LD_LIBRARY_PATH="/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-online}"
unset WANDB_DISABLED
CIF_PY="${CIF_PYTHON:-/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python}"
CIF_WEIGHT="${CIF_CTC_WEIGHT:-0}"
CIF_OTTC_WEIGHT="${CIF_OTTC_WEIGHT:-0}"
CIF_NAME="no_ctc"
if [[ "$CIF_WEIGHT" != 0 ]]; then CIF_NAME="ctc_${CIF_WEIGHT//./p}"; fi
if [[ "$CIF_OTTC_WEIGHT" != 0 ]]; then CIF_NAME="ottc_jsd_${CIF_OTTC_WEIGHT//./p}"; fi
CIF_DEFAULT_OUTPUT="$PWD/reproduction/cif_timit_20261016/runs/${CIF_NAME}_seed42"
CIF_ACTIVE="$PWD/reproduction/cif_timit_20261016/active_runs.json"
if [[ -z "${CIF_OUTPUT:-}" && -f "$CIF_ACTIVE" ]]; then
  CIF_DEFAULT_OUTPUT=$("$CIF_PY" - "$CIF_ACTIVE" "$CIF_NAME" "$CIF_DEFAULT_OUTPUT" <<'PY'
import json,sys
from pathlib import Path
runs=json.loads(Path(sys.argv[1]).read_text())
print(runs.get(sys.argv[2], sys.argv[3]))
PY
)
fi
exec "$CIF_PY" -u -m reproduction.cif_timit_20261016.train \
  --output "${CIF_OUTPUT:-$CIF_DEFAULT_OUTPUT}" \
  --ctc-weight "$CIF_WEIGHT" --ottc-weight "$CIF_OTTC_WEIGHT" \
  --max-wall-seconds "${CIF_MAX_WALL_SECONDS:-3000}" "$@"
