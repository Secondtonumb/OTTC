#!/usr/bin/env bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
export PYTHONPATH="$PWD/reproduction/vendor:$PWD"
export LD_LIBRARY_PATH="/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-online}"
unset WANDB_DISABLED
export CIF_PY=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python
export CIF_Q_ROOT="$PWD/reproduction/cif_quantity_20261016"
