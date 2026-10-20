#!/usr/bin/env bash
set -euo pipefail
task_root=/work/gm64/m64000/OTTC/reproduction/mfa_direct_20261017
export PATH="$task_root/runtime/bin:$PATH"
export LD_LIBRARY_PATH="$task_root/runtime/lib:${LD_LIBRARY_PATH:-}"
export MFA_ROOT_DIR="$task_root/mfa_state"
export MPLCONFIGDIR="$task_root/matplotlib_cache"
export NUMBA_CACHE_DIR="$task_root/numba_cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
python "$task_root/align_api.py"
