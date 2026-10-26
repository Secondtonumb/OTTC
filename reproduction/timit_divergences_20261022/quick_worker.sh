#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline OTTC_EVAL_UPLOAD=0
export WANDB_CACHE_DIR="$QUICK_ROOT/wandb_cache" WANDB_DATA_DIR="$QUICK_ROOT/wandb_data"
quick_rank=${OMPI_COMM_WORLD_RANK:?}
exec "$LIBRI_PYTHON" -u -m torch.distributed.run \
    --nnodes 8 --nproc_per_node 1 --node_rank "$quick_rank" \
    --master_addr "$QUICK_MASTER" --master_port "$QUICK_PORT" \
    --module reproduction.timit_divergences_20261022.quick_suite \
    --epochs 10 --deadline "$QUICK_DEADLINE" --output-dir "$QUICK_ROOT" \
    > "$QUICK_ROOT/rank-$quick_rank.log" 2>&1
