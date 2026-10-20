#!/bin/bash
export PYTHONPATH=/work/gm64/m64000/OTTC/reproduction/vendor:/work/gm64/m64000/OTTC:/work/gm64/m64000/zipa/mfa_miyabi_20260922/python_site
export LD_LIBRARY_PATH=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1 TORCHDYNAMO_DISABLE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=${WANDB_MODE:-online} WANDB_LOG_MODEL=false WANDB_WATCH=false
export WANDB_RUN_GROUP=libri960_continue5_20261017
unset WANDB_DISABLED
LIBRI_PYTHON=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python

export TOKENIZERS_PARALLELISM=false
# Only repository-owned trusted Trainer/optimizer/RNG checkpoints are loaded.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
