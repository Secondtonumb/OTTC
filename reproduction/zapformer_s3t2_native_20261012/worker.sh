#!/bin/bash
set -euo pipefail
task_dir=/work/gm64/m64000/OTTC/reproduction/zapformer_s3t2_native_20261012
source "$task_dir/environment.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
task_args=()
if [[ "$S3T2_STAGE" == framece ]]; then
  task_args+=(--init-from "${S3T2_INIT_CHECKPOINT:?Set S3T2_INIT_CHECKPOINT to an ordinary S3T2 JSD checkpoint}")
else
  # This legacy entry remains the MFA-full experiment, not the new Libri100 run.
  task_args+=(--jsd-manifest /work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/training.json)
fi
"$MFA_PYTHON" -m torch.distributed.run --nnodes "${S3T2_NODES:?}" --nproc_per_node 1 \
  --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${S3T2_MASTER_ADDR:?}" \
  --master_port "${S3T2_MASTER_PORT:?}" "$task_dir/train_s3t2.py" \
  --objective "$S3T2_STAGE" --output "$task_dir/training_$S3T2_STAGE" \
  --epochs "$S3T2_EPOCHS" --base-lr "${S3T2_LR:-.02}" --max-duration "${S3T2_MAX_DURATION:-100}" \
  --save-every 50 --eval-every 50 --average-period 200 --max-wall-seconds 13800 "${task_args[@]}"
