#!/bin/bash
export RECIPE_ROOT=/work/gm64/m64000/OTTC/reproduction/english_suite/zapformer_recipe_s3t2_20261010
export RECIPE_PYTHON=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python
export LD_LIBRARY_PATH=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$RECIPE_ROOT/vendor/zapformer:$RECIPE_ROOT/vendor:$RECIPE_ROOT:/work/gm64/m64000/zipa/mfa_miyabi_20260922/python_site:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none
export WANDB_PROJECT=${WANDB_PROJECT:-$("$RECIPE_PYTHON" -c 'import json; print(json.load(open("/work/gm64/m64000/OTTC/reproduction/wandb_config.json"))["project"])')}
export WANDB_RUN_GROUP=zapformer_recipe_s3t2_20261010
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.75,roundup_power2_divisions:8
