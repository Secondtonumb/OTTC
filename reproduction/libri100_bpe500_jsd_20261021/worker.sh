#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
export CUDA_VISIBLE_DEVICES=0 OTTC_EVAL_UPLOAD=0
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none
exec "$LIBRI_PYTHON" -m torch.distributed.run --nnodes=2 --nproc_per_node=1 \
    --node_rank "${OMPI_COMM_WORLD_RANK:?MPI rank required}" \
    --master_addr "${LIBRI_MASTER:?Master node required}" --master_port "${LIBRI_PORT:?Master port required}" \
    reproduction/libri100_bpe500_jsd_20261021/train.py "$@"