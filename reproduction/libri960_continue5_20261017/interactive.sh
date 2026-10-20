#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
export LIBRI960_DEADLINE=$(($(date +%s)+5700))
"$LIBRI_PYTHON" -u reproduction/check_js_cost.py
for objective in jsd_upstream; do
    bash reproduction/libri960_continue5_20261017/worker.sh "$objective" validate \
        >> "reproduction/libri960_continue5_20261017/validate_${objective}.log" 2>&1
done
status=0
bash reproduction/libri960_continue5_20261017/worker.sh jsd_upstream train \
    >> reproduction/libri960_continue5_20261017/train_jsd_upstream.log 2>&1 || status=$?
printf '%s\n' "$status" > reproduction/libri960_continue5_20261017/interactive_exit_code.txt
exit "$status"
