#!/bin/bash
set -euo pipefail
export MFA_PYTHON=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python
export PYTHONPATH=/work/gm64/m64000/zipa/mfa_miyabi_20260922/python_site:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}
export MFA_TMPDIR=/tmp
source /work/gm64/m64000/zipa/mfa_miyabi_20260922/zipa/experiments/mfa_zapformer/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
