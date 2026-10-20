#!/usr/bin/env bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
test -f reproduction/multihead_timit/validation/GPU_SMOKE_COMPLETE
export PYTHONPATH=/work/gm64/m64000/OTTC/reproduction/vendor:/work/gm64/m64000/OTTC
export LD_LIBRARY_PATH=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false
for configuration_rank in 0 1 2; do
  OMPI_COMM_WORLD_RANK="$configuration_rank" bash reproduction/multihead_timit/train_rank.sh
done
/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python reproduction/multihead_timit/summarize.py
# Results and W&B uploads have finished; release the interactive GPU promptly.
# The dependency fallback only summarizes completed runs if already released.
if [[ -n "${PBS_JOBID:-}" ]]; then
  qdel "$PBS_JOBID"
fi
