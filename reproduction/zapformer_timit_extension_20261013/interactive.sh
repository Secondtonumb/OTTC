#!/usr/bin/env bash
# Run after the original 40-epoch command finishes, inside a GPU allocation.
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/zapformer_s3t2_native_20261012/environment.sh
task_dir=/work/gm64/m64000/OTTC/reproduction/zapformer_timit_extension_20261013
task_epochs="${TIMIT_EPOCHS:-80}"
task_parent="${TIMIT_PARENT_OUTPUT:-/work/gm64/m64000/OTTC/reproduction/zapformer_timit_20261013/runs/jsd_seed42}"
task_output="${TIMIT_OUTPUT:-$task_dir/runs/jsd_seed42_to${task_epochs}}"
task_budget="${TIMIT_MAX_WALL_SECONDS:-3600}"
for task_arg in "$@"; do
  case "$task_arg" in
    --output|--output=*|--parent-output|--parent-output=*|--epochs|--epochs=*|--cost|--cost=*|--data|--data=*)
      echo 'Use TIMIT_OUTPUT/TIMIT_PARENT_OUTPUT/TIMIT_EPOCHS; objective and data are fixed.' >&2; exit 2 ;;
  esac
done
[[ -f "$task_parent/TRAIN_COMPLETE.json" ]] || { echo 'Finish the original run first.' >&2; exit 2; }
"$MFA_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Use an interactive GPU node"'
mkdir -p "$task_output"
exec 9> "$task_output/.training.lock"
flock -n 9 || { echo 'This extension already has an active training process' >&2; exit 1; }
exec 8> "$task_parent/.training.lock"
flock -n 8 || { echo 'Wait for the original training process to exit' >&2; exit 1; }
export PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-online}"
"$MFA_PYTHON" "$task_dir/train_timit.py" \
  --parent-output "$task_parent" --epochs "$task_epochs" \
  --data /work/gm64/m64000/OTTC/reproduction/zapformer_timit_20261013/data \
  --cost jsd --output "$task_output" --max-duration 100 --base-lr .02 \
  --max-wall-seconds "$task_budget" "$@" 2>&1 | tee -a "$task_output/interactive.log"
