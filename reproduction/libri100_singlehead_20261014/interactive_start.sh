#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/libri100_singlehead_20261014/environment.sh
export LIBRI_DEADLINE=$(($(date +%s)+5700))
# Data is being prepared on the login node; avoid a competing cache writer.
while [[ ! -f reproduction/libri100_singlehead_20261014/data/PREPARE_COMPLETE.json ]]; do
    if (( $(date +%s) >= LIBRI_DEADLINE-600 )); then exit 75; fi
    sleep 5
done
"$LIBRI_PYTHON" -u reproduction/check_js_cost.py
for objective in ce jsd_upstream jsd_standard; do
    bash reproduction/libri100_singlehead_20261014/worker.sh "$objective" validate 0 \
        >> "reproduction/libri100_singlehead_20261014/validate_${objective}.log" 2>&1
done
"$LIBRI_PYTHON" -u reproduction/libri100_singlehead_20261014/gate.py
"$LIBRI_PYTHON" reproduction/libri100_singlehead_20261014/update_status.py 'three-cost GPU/K2 validation passed; interactive upstream JSD starting; parallel continuation accepted'
# Prioritize the user's original-upstream JSD while the batch suite is queued.
status=0
bash reproduction/libri100_singlehead_20261014/worker.sh jsd_upstream train 0 \
    >> reproduction/libri100_singlehead_20261014/train_jsd_upstream.log 2>&1 || status=$?
printf '%s\n' "$status" > reproduction/libri100_singlehead_20261014/interactive_exit_code.txt
"$LIBRI_PYTHON" reproduction/libri100_singlehead_20261014/update_status.py 'interactive phase ended; three-node continuation pending'
exit "$status"
