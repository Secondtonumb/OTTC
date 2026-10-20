#!/bin/bash
set -euo pipefail
task_dir=/work/gm64/m64000/OTTC/reproduction/spottc_jsd_s3t2_framece_libri100_20261012
source "$task_dir/environment.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
"$MFA_PYTHON" -m torch.distributed.run --nnodes 4 --nproc_per_node 1 \
  --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${FRAMECE_MASTER_ADDR:?}" \
  --master_port "${FRAMECE_MASTER_PORT:?}" "$task_dir/train_framece.py" \
  --output "$task_dir/training_user_replay" --epochs 3 --batch-size 32 --lr .0001 \
  --save-every 50 --eval-every 50 --max-wall-seconds 13800
