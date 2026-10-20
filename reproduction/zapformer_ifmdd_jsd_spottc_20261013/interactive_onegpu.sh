#!/bin/bash
# Run inside a Miyabi interactive GPU allocation. Does not submit another job.
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$task_dir/../.." && pwd)
cd "$repo_dir"
source "$repo_dir/reproduction/zapformer_s3t2ss_native_20261012/environment.sh"

export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
# Keep the scheduler's assigned GPU; an explicit GPU_DEVICE overrides it.
if [[ -n "${GPU_DEVICE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
else
  export CUDA_VISIBLE_DEVICES=0
fi
test_mode="${TEST_MODE:-full}"
case "$test_mode" in full|pilot|parity) ;; *) echo 'TEST_MODE must be full, pilot or parity' >&2; exit 2 ;; esac
test_output="${TEST_OUTPUT:-$task_dir/interactive/$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$test_output"
test_output=$(cd -- "$test_output" && pwd)
exec 9> "$test_output/.test.lock"
flock -n 9 || { echo "Test is already running: $test_output" >&2; exit 1; }
exec > >(tee -a "$test_output/console.log") 2>&1
printf 'IF-MDD standard-JSD SPOTTC; mode=%s; output=%s\n' "$test_mode" "$test_output"

if [[ "$test_mode" == parity ]]; then
  "$MFA_PYTHON" "$task_dir/check_parity.py" --output "$test_output/PARITY_PASS.json" --device cpu
else
  "$MFA_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Run this script inside a GPU allocation; use TEST_MODE=parity for CPU checks.')
assert torch.cuda.device_count()==1, 'Expected exactly one visible GPU'
print('GPU:',torch.cuda.get_device_name(0),'; PyTorch:',torch.__version__,flush=True)
PY
  test_args=(--output-root "$test_output" --overfit-steps "${OVERFIT_STEPS:-200}")
  if [[ "$test_mode" == pilot ]]; then test_args+=(--pilot-only); fi
  "$MFA_PYTHON" "$task_dir/validate.py" "${test_args[@]}"
  "$MFA_PYTHON" "$task_dir/summarize.py" --output-root "$test_output"
  "$MFA_PYTHON" "$task_dir/publish.py" --output-root "$test_output"
fi
printf 'Completed. Results: %s\n' "$test_output"
