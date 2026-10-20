#!/usr/bin/env bash
# Run this AFTER qsub -I grants a GPU allocation; never on a login node.
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/zapformer_s3t2_native_20261012/environment.sh
task_dir=/work/gm64/m64000/OTTC/reproduction/zapformer_timit_20261013
task_cost="${TIMIT_COST:-jsd}"
if [[ "$task_cost" != jsd && "$task_cost" != ce ]]; then
  echo 'TIMIT_COST must be jsd or ce' >&2; exit 2
fi
task_output="${TIMIT_OUTPUT:-$task_dir/runs/${task_cost}_seed42}"
for task_arg in "$@"; do
  case "$task_arg" in
    --output|--output=*|--cost|--cost=*|--data|--data=*)
      echo 'Set TIMIT_OUTPUT/TIMIT_COST via environment; the cached data path is fixed' >&2
      exit 2 ;;
  esac
done
"$MFA_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Request a GPU allocation with qsub -I first"'
mkdir -p "$task_output"
exec 9> "$task_output/.training.lock"
flock -n 9 || { echo 'This output already has an active training process' >&2; exit 1; }
# Forwarded flags can select a tiny/short validation or change hyperparameters.
# Existing runs reject changed training contracts; choose TIMIT_OUTPUT for a new run.
"$MFA_PYTHON" "$task_dir/prepare.py" --output "$task_dir/data"
export PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-online}"
"$MFA_PYTHON" "$task_dir/train_timit.py" \
  --data "$task_dir/data" --cost "$task_cost" --output "$task_output" \
  --epochs "${TIMIT_EPOCHS:-40}" --max-duration "${TIMIT_MAX_DURATION:-100}" \
  --base-lr "${TIMIT_LR:-0.02}" --max-wall-seconds "${TIMIT_MAX_WALL_SECONDS:-10200}" \
  "$@" 2>&1 | tee -a "$task_output/interactive.log"
