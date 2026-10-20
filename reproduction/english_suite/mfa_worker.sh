#!/usr/bin/env bash
set -euo pipefail
base=/work/gm64/m64000/zipa/mfa_miyabi_20260922
export MFA_REPO="$base/zipa" MFA_HANDOFF="$base/handoff_local"
export MFA_DATA=/work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/training.json
export MFA_PYTHON=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python
export PYTHONPATH="$base/python_site:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}
export MFA_TMPDIR=/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=online WANDB_LOG_MODEL=false WANDB_WATCH=false
mkdir -p "${MFA_OUTPUT:?}"
cd "$MFA_REPO"
source experiments/mfa_zapformer/env.sh
arm="${1:?}"; shift
case "$arm" in
  ctc|ottc) cap=315; initial_step=000005253;;
  spottc) cap=320; initial_step=000003444;;
  film_k4) cap=200; initial_step=;;
  *) exit 2;;
esac
arguments=(--arm "$arm" --data "$MFA_DATA" --output "$MFA_OUTPUT" --max-duration "$cap" --num-epochs 10 --lr .02 --saves-per-epoch 4 --max-wall-seconds 28200)
if [[ "${MFA_USE_LOCAL_RESUME:-0}" == 1 ]]; then
  latest=$(find "$MFA_OUTPUT" -maxdepth 1 -name 'step-*.pt' | sort | tail -1)
else
  latest=
fi
if [[ -n "$latest" ]]; then
  arguments+=(--resume "$latest")
elif [[ -n "$initial_step" ]]; then
  arguments+=(--resume "$MFA_HANDOFF/checkpoints/$arm/step-$initial_step.pt" --migration-contract "$MFA_REPO/experiments/mfa_zapformer/runs/full_8h_20260922/$arm/run_config.json")
fi
tracking_entry=/work/gm64/m64000/OTTC/reproduction/english_suite/mfa_tracked.py
if [[ "${MFA_OPTIMIZE_MICROBATCH:-0}" == 1 ]]; then
  tracking_entry=/work/gm64/m64000/OTTC/reproduction/english_suite/mfa_optimized.py
fi
exec "$MFA_PYTHON" -m torch.distributed.run --nnodes 8 --nproc_per_node 1 \
  --node_rank "${OMPI_COMM_WORLD_RANK:?}" --master_addr "${MFA_MASTER_ADDR:?}" --master_port "${MFA_MASTER_PORT:?}" \
  "$tracking_entry" "${arguments[@]}" "$@"
