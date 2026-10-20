#!/usr/bin/env bash
set -euo pipefail
base=/work/gm64/m64000/zipa/mfa_miyabi_20260922
export MFA_REPO="$base/zipa"
export MFA_PYTHON=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python
export PYTHONPATH="$base/python_site:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}
export MFA_TMPDIR=/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false
cd "$MFA_REPO"
source experiments/mfa_zapformer/env.sh
arguments=(--arm ottc --data /work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/training.json
  --output "${MFA_JSD_OUTPUT:?}" --max-duration 315 --num-epochs 10 --lr .02
  --saves-per-epoch 4 --max-wall-seconds 28200)
latest=$(find "$MFA_JSD_OUTPUT" -maxdepth 1 -name 'step-*.pt' | sort | tail -1)
if [[ -n "$latest" ]]; then arguments+=(--resume "$latest"); fi
exec "$MFA_PYTHON" -m torch.distributed.run --nnodes 8 --nproc_per_node 1 \
  --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${MFA_MASTER_ADDR:?}" --master_port "${MFA_MASTER_PORT:?}" \
  /work/gm64/m64000/OTTC/reproduction/english_suite/mfa_jsd/train_jsd.py "${arguments[@]}"
