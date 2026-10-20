#!/usr/bin/env bash
set -euo pipefail
source /work/gm64/m64000/OTTC/reproduction/cif_quantity_20261016/environment.sh
CIF_DATASET="${1:?buckeye or libri100}"
CIF_VARIANT="${2:?ce, ctc_0p5 or ottc_jsd_0p5}"
"$CIF_PY" - "$CIF_DATASET" "$CIF_VARIANT" <<'PY'
import json,os,sys
from pathlib import Path
from reproduction.cif_quantity_20261016.data import ROOT,sha,load_rows
r=Path(os.environ['CIF_Q_ROOT']);g=json.loads((r/'GPU_PROFILE_PASS.json').read_text())
assert g['quantity_weight']==1 and g['batch_size']==64
profile=g['profiles'][sys.argv[1]+'_'+sys.argv[2]]['config']
for f,h in profile['sources_sha256'].items():assert sha(ROOT/f)==h,f
_,_,data=load_rows(sys.argv[1],1,1)
assert data['manifests_sha256']==profile['manifests_sha256']
assert data['vocab_sha256']==profile['vocab_sha256']
assert data['shared_data_sha256']==profile['shared_data_sha256']
print('QUANTITY=1 PROFILE, SOURCE, DATA GATE PASS',flush=True)
PY
CIF_ARGS=()
case "$CIF_VARIANT" in
  ce) ;;
  ctc_0p5) CIF_ARGS=(--ctc-weight 0.5) ;;
  ottc_jsd_0p5) CIF_ARGS=(--ottc-weight 0.5) ;;
  *) exit 2 ;;
esac
mkdir -p "$CIF_Q_ROOT/runs"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 2 > "$CIF_Q_ROOT/runs/${CIF_DATASET}_${CIF_VARIANT}_${PBS_JOBID}.telemetry.csv" &
CIF_TELEMETRY=$!
trap 'kill "$CIF_TELEMETRY" 2>/dev/null || true' EXIT
"$CIF_PY" -u -m reproduction.cif_quantity_20261016.train \
  --dataset "$CIF_DATASET" --output "$CIF_Q_ROOT/runs/${CIF_DATASET}_${CIF_VARIANT}_q1_seed42" \
  --quantity-weight 1 "${CIF_ARGS[@]}" --epochs 40 --batch-size 64 \
  --max-wall-seconds "${CIF_Q_MAX_WALL_SECONDS:-6600}"
