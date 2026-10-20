#!/usr/bin/env bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
CIF_NQ_END=$(date -d '2026-10-16 19:40:46 +0900' +%s)
CIF_NQ_NOW=$(date +%s)
export CIF_NQ_MAX_WALL_SECONDS=$((CIF_NQ_END-CIF_NQ_NOW-300))
if (( CIF_NQ_MAX_WALL_SECONDS < 120 )); then exit 0; fi
bash reproduction/cif_noquantity_20261016/worker.sh libri100 ottc_jsd_0p5
