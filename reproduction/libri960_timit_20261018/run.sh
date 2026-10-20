#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_GROUP=libri960_timit_20261018
"$LIBRI_PYTHON" -u reproduction/libri960_timit_20261018/evaluate.py --smoke --objectives jsd_upstream --variants libri100_epoch40
"$LIBRI_PYTHON" -u reproduction/libri960_timit_20261018/evaluate.py
"$LIBRI_PYTHON" -u reproduction/libri960_timit_20261018/summarize.py
