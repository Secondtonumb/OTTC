#!/bin/bash
set -euo pipefail
source /work/gm64/m64000/OTTC/reproduction/english_suite/zapformer_recipe_s3t2_20261010/env.sh
export CUDA_VISIBLE_DEVICES=0
arguments=(--arm spottc --data /work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/training.json
  --output "$RECIPE_ROOT/training" --max-duration 320 --num-epochs 10 --lr .02
  --saves-per-epoch 4 --max-wall-seconds 27000)
latest_checkpoint=""
while IFS= read -r checkpoint_path; do latest_checkpoint="$checkpoint_path"; done < <(
  rg --files "$RECIPE_ROOT/training" -g 'step-*.pt' | sort)
if [[ -n "$latest_checkpoint" ]]; then arguments+=(--resume "$latest_checkpoint"); fi
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 10 > "$RECIPE_ROOT/training/gpu_rank${OMPI_COMM_WORLD_RANK}_${PBS_JOBID}.csv" &
telemetry_pid=$!
trap 'kill "$telemetry_pid" 2>/dev/null || true' EXIT
"$RECIPE_PYTHON" -m torch.distributed.run --nnodes 8 --nproc_per_node 1 \
  --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${RECIPE_MASTER_ADDR:?}" --master_port "${RECIPE_MASTER_PORT:?}" \
  "$RECIPE_ROOT/train.py" "${arguments[@]}"

