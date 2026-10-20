#!/usr/bin/env bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
CIF_Q_END=$(date -d '2026-10-16 20:55:39 +0900' +%s)
CIF_Q_NOW=$(date +%s)
export CIF_Q_MAX_WALL_SECONDS=$((CIF_Q_END-CIF_Q_NOW-300))
if (( CIF_Q_MAX_WALL_SECONDS < 120 )); then exit 0; fi
bash reproduction/cif_quantity_20261016/worker.sh libri100 ottc_jsd_0p5
