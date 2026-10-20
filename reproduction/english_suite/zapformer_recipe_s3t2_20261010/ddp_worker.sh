#!/bin/bash
set -euo pipefail
source /work/gm64/m64000/OTTC/reproduction/english_suite/zapformer_recipe_s3t2_20261010/env.sh
export CUDA_VISIBLE_DEVICES=0
arguments=(--nnodes 8 --nproc_per_node 1 --node_rank "${OMPI_COMM_WORLD_RANK:?}"
 --master_addr "${RECIPE_MASTER_ADDR:?}" --master_port "${RECIPE_MASTER_PORT:?}")
"$RECIPE_PYTHON" -m torch.distributed.run "${arguments[@]}" "$RECIPE_ROOT/validate.py" --ddp
"$RECIPE_PYTHON" -m torch.distributed.run "${arguments[@]}" "$RECIPE_ROOT/validate.py" --ddp --resume-phase

