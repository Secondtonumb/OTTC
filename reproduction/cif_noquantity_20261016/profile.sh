#!/usr/bin/env bash
set -euo pipefail
source /work/gm64/m64000/OTTC/reproduction/cif_noquantity_20261016/environment.sh
export CIF_NQ_PROBE="${CIF_NQ_PROBE:-$CIF_NQ_ROOT/probes/${PBS_JOBID:?Run on allocated GPU}}"
mkdir -p "$CIF_NQ_PROBE"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -l 1 > "$CIF_NQ_PROBE/telemetry.csv" &
CIF_TELEMETRY=$!
trap 'kill "$CIF_TELEMETRY" 2>/dev/null || true' EXIT
"$CIF_PY" -m unittest reproduction.cif_timit_20261016.test_cif reproduction.cif_noquantity_20261016.test_recipe
for CIF_DATASET in buckeye libri100; do
  for CIF_VARIANT in ce ctc_0p5 ottc_jsd_0p5; do
    CIF_ARGS=()
    case "$CIF_VARIANT" in
      ctc_0p5) CIF_ARGS=(--ctc-weight 0.5) ;;
      ottc_jsd_0p5) CIF_ARGS=(--ottc-weight 0.5) ;;
    esac
    "$CIF_PY" -u -m reproduction.cif_noquantity_20261016.train \
      --dataset "$CIF_DATASET" --output "$CIF_NQ_PROBE/${CIF_DATASET}_${CIF_VARIANT}" \
      --quantity-weight 0 "${CIF_ARGS[@]}" --profile --max-steps 4 --batch-size 64 \
      --eval-limit 16 --evaluate-on-stop --max-wall-seconds 900
  done
done
"$CIF_PY" -u -m reproduction.cif_noquantity_20261016.check_profile
