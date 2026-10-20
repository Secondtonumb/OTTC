#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/zapformer_s3t2_native_20261012/environment.sh
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline
task_dir=/work/gm64/m64000/OTTC/reproduction/zapformer_gpu_profile_20261012
"$MFA_PYTHON" -m torch.distributed.run --nnodes "${NATIVE_PROFILE_NODES:?}" --nproc_per_node=1 \
  --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${NATIVE_PROFILE_MASTER:?}" --master_port "${NATIVE_PROFILE_PORT:?}" \
  "$task_dir/native_gpu_probe.py" --output "${NATIVE_PROFILE_OUTPUT:?}" --max-duration "${NATIVE_PROFILE_DURATION:?}"
