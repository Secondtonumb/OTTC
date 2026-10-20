#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
task_native=/work/gm64/m64000/OTTC/reproduction/zapformer_s3t2ss_native_20261012
task_stage=/work/gm64/m64000/OTTC/reproduction/zapformer_libri100_s3t2ss_native_20261012
source "$task_native/environment.sh"
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE="${WANDB_MODE:-online}"
"$MFA_PYTHON" -m torch.distributed.run --nnodes 8 --nproc_per_node=1 \
  --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${NATIVE_TRAIN_MASTER:?}" --master_port "${NATIVE_TRAIN_PORT:?}" \
  "$task_native/train_s3t2.py" --objective jsd --topology s3_t2_star_star --output "$task_stage/training" \
  --jsd-manifest /work/gm64/m64000/OTTC/reproduction/zapformer_libri100_data_20261012/training.json \
  --epochs 10 --base-lr .02 --max-duration 640 --save-every 50 --eval-every 50 --average-period 200 \
  --max-wall-seconds "${NATIVE_TASK_WALL_SECONDS:-13800}"
