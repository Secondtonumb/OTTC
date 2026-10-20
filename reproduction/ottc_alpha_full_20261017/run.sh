#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
export PYTHONPATH="$PWD/reproduction/vendor:$PWD"
export LD_LIBRARY_PATH="/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 >> reproduction/ottc_alpha_full_20261017/telemetry.csv &
alpha_full_telemetry=$!
trap 'kill "$alpha_full_telemetry" 2>/dev/null || true' EXIT
/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python -u -m reproduction.ottc_alpha_full_20261017.decode
