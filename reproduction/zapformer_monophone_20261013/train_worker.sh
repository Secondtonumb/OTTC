#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
mono_stage=/work/gm64/m64000/OTTC/reproduction/zapformer_monophone_20261013
"$MFA_PYTHON" -m torch.distributed.run --nnodes 8 --nproc_per_node 1 \
  --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${MONO_MASTER:?}" --master_port "${MONO_PORT:?}" \
  "$mono_stage/train_native.py" --objective jsd --topology s1_t1 --output "$mono_stage/training" \
  --epochs "${MONO_EPOCHS:-10}" --base-lr .03423791569515573 --max-duration 640 \
  --save-every 50 --eval-every 50 --average-period 200 --keep-last 8 --max-wall-seconds 13800
