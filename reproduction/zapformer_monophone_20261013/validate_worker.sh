#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
export WANDB_MODE=offline
for mono_pass in uninterrupted resumed_first resumed_second; do
  mono_steps=2
  mono_run=$mono_pass
  if [[ "$mono_pass" == resumed_first ]]; then mono_steps=1; mono_run=resumed; fi
  if [[ "$mono_pass" == resumed_second ]]; then mono_run=resumed; fi
  "$MFA_PYTHON" -m torch.distributed.run --nnodes 8 --nproc_per_node 1 \
    --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${MONO_MASTER:?}" --master_port "${MONO_PORT:?}" \
    reproduction/zapformer_monophone_20261013/train_native.py --objective jsd --topology s1_t1 \
    --output "/work/gm64/m64000/OTTC/reproduction/zapformer_monophone_20261013/validation_ddp/$mono_run" \
    --smoke --epochs 10 --base-lr .03423791569515573 --stop-after "$mono_steps" --max-duration 100 \
    --save-every 1 --eval-every 1 --average-period 2 --eval-limit 8
done
