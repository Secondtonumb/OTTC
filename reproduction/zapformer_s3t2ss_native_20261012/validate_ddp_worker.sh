#!/bin/bash
set -euo pipefail
task_dir=/work/gm64/m64000/OTTC/reproduction/zapformer_s3t2ss_native_20261012
source "$task_dir/environment.sh"
export WANDB_MODE=offline
cd /work/gm64/m64000/OTTC
for task_objective in ${S3T2_VALIDATION_OBJECTIVES:-jsd}; do
  for task_pass in uninterrupted resumed_first resumed_second; do
    task_steps=2
    task_run=$task_pass
    if [[ "$task_pass" == resumed_first ]]; then task_steps=1; task_run=resumed; fi
    if [[ "$task_pass" == resumed_second ]]; then task_run=resumed; fi
    "$MFA_PYTHON" -m torch.distributed.run --nnodes "${S3T2_TEST_NODES:-4}" --nproc_per_node 1 \
      --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${S3T2_MASTER_ADDR:?}" --master_port "${S3T2_MASTER_PORT:?}" \
      "$task_dir/train_s3t2.py" --objective "$task_objective" --output "${S3T2_VALIDATION_ROOT:-$task_dir/validation_ddp}/$task_objective/$task_run" \
      --topology s3_t2_star_star --tiny --smoke --stop-after "$task_steps" --max-duration 100 --save-every 1 --eval-every 1 --average-period 2 --eval-limit 8
  done
done
