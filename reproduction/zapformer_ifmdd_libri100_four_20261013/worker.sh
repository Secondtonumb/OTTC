#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
task_stage=/work/gm64/m64000/OTTC/reproduction/zapformer_ifmdd_libri100_four_20261013
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE="${WANDB_MODE:-online}"
task_duration="$("$MFA_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_duration"])' "$task_stage/BATCH_SELECTION.json")"
task_extra=()
if [[ -n "${SP4_STOP_AFTER:-}" ]]; then
 task_extra+=(--stop-after "$SP4_STOP_AFTER" --save-every 1 --eval-every 1000 --average-period 2 --keep-last 0 --eval-limit 8 --skip-initial-eval)
else
 task_extra+=(--save-every 50 --eval-every 50 --average-period 200 --keep-last 8)
fi
"$MFA_PYTHON" -m torch.distributed.run --nnodes 8 --nproc_per_node=1 \
 --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${SP4_MASTER:?}" --master_port "${SP4_PORT:?}" \
 "$task_stage/train_ottc.py" --arm "${SPOTTC_ARM:?}" --output "${SP4_OUTPUT:?}" \
 --epochs 50 --base-lr .02 --encoder-aux-loss-scale .05 --max-duration "$task_duration" \
 --max-wall-seconds "${SP4_WALL_SECONDS:-13800}" "${task_extra[@]}"
