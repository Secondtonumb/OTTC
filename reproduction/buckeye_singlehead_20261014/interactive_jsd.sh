#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/buckeye_singlehead_20261014/environment.sh
exec 9> reproduction/buckeye_singlehead_20261014/.jsd_upstream.lock
flock -n 9
set +e
"$BUCKEYE_PYTHON" -u reproduction/buckeye_singlehead_20261014/budget_train.py --objective jsd_upstream --max-wall-seconds 5900
train_status=$?
set -e
if [[ "$train_status" == 75 ]]; then
    exit 75
fi
if [[ "$train_status" != 0 ]]; then
    exit "$train_status"
fi
"$BUCKEYE_PYTHON" -u reproduction/buckeye_singlehead_20261014/evaluate.py --objective jsd_upstream
