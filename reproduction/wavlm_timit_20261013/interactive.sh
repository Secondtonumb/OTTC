#!/usr/bin/env bash
# Invoke only inside an allocated single-GPU node. Repeated calls resume.
set -euo pipefail
cd /work/gm64/m64000/OTTC
task_dir=/work/gm64/m64000/OTTC/reproduction/wavlm_timit_20261013
task_result=/work/gm64/m64000/OTTC/reproduction/results/runexp_ottc_fuse_phoneme_jsd_wavlm_20261013
task_python=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python
task_budget="${WAVLM_MAX_WALL_SECONDS:-3600}"
[[ "$task_budget" =~ ^[0-9]+$ && "$task_budget" -gt 300 ]] || { echo 'WAVLM_MAX_WALL_SECONDS must be an integer greater than 300'; exit 2; }
task_started=$(date +%s)
export PYTHONPATH=/work/gm64/m64000/OTTC/reproduction/vendor:/work/gm64/m64000/OTTC
export LD_LIBRARY_PATH=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
unset WANDB_DISABLED
export WANDB_MODE="${WANDB_MODE:-online}" WANDB_LOG_MODEL=false WANDB_WATCH=false
export WANDB_RUN_GROUP=wavlm_timit_singlehead_20261013
"$task_python" -c 'import torch; assert torch.cuda.is_available(), "Run inside an allocated GPU node"; assert torch.cuda.device_count()==1, "Expose one GPU using CUDA_VISIBLE_DEVICES"; print(torch.cuda.get_device_name())'
if [[ -f "$task_result/EVAL_COMPLETE.json" ]]; then
  echo "Already trained/evaluated: $task_result/metrics.json"; exit 0
fi
exec 9> "$task_dir/.job.lock"
flock -n 9 || { echo 'This WavLM experiment already has an active process'; exit 2; }
exec > >(tee -a "$task_dir/interactive.log") 2>&1
echo "Interactive WavLM Large upstream JSD; total epochs=40; invocation budget=$task_budget seconds"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1 >> "$task_dir/interactive_telemetry.csv" &
task_monitor_pid=$!
trap 'kill "$task_monitor_pid" 2>/dev/null || true' EXIT
if [[ ! -f "$task_result/TRAIN_COMPLETE.json" ]]; then
  if ! "$task_python" - <<'PY'
import sys
sys.path.insert(0,'reproduction/wavlm_timit_20261013')
from bootstrap import HERE,source_hashes
import json
path=HERE/'profile/PASS.json'
if not path.exists() or json.loads(path.read_text())['source_sha256']!=source_hashes():
    print('Preparing exact-recipe GPU profile',flush=True)
    raise SystemExit(1)
PY
  then
    "$task_python" -u "$task_dir/gpu_probe.py"
  fi
  task_elapsed=$(( $(date +%s) - task_started ))
  task_remaining=$(( task_budget - task_elapsed - 300 ))
  if (( task_remaining <= 0 )); then
    echo 'Profile completed; insufficient budget for training. Rerun in a fresh allocation.'; exit 0
  fi
  "$task_python" -u "$task_dir/launch.py" --max-wall-seconds "$task_remaining"
fi
if [[ -f "$task_result/TRAIN_COMPLETE.json" ]]; then
  "$task_python" -u "$task_dir/evaluate.py" --loss ottc --bins-metric jsd --cross-domain
else
  echo 'Checkpoint saved. Run this same script again to continue to total 40 epochs.'
fi
