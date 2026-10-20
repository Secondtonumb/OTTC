#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
task_stage=/work/gm64/m64000/OTTC/reproduction/zapformer_libri100_jsd50_native_20261013
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE="${WANDB_MODE:-online}"
task_extra=()
if [[ -n "${JSD50_STOP_AFTER:-}" ]]; then task_extra+=(--stop-after "$JSD50_STOP_AFTER" --eval-limit 8); fi
"$MFA_PYTHON" -m torch.distributed.run --nnodes 8 --nproc_per_node=1 \
  --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${JSD50_MASTER:?}" --master_port "${JSD50_PORT:?}" \
  "$task_stage/continue_jsd.py" --continuation-spec "${JSD50_OUTPUT:?}/continuation_spec.json" \
  --objective jsd --topology s3_t2_star_star --output "$JSD50_OUTPUT" \
  --jsd-manifest /work/gm64/m64000/OTTC/reproduction/zapformer_libri100_data_20261012/training.json \
  --epochs 50 --base-lr .02 --max-duration 640 --save-every 50 --eval-every 50 --average-period 200 \
  --max-wall-seconds "${JSD50_WALL_SECONDS:-13800}" "${task_extra[@]}"
