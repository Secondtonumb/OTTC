#!/usr/bin/env bash
set -euo pipefail
source /work/gm64/m64000/OTTC/reproduction/ottc_transducer_timit_20261017/environment.sh
exec "$OTTC_TRANS_PY" -u -m reproduction.ottc_transducer_timit_20261017.train "$@"
