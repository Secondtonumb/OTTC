#!/usr/bin/env bash
set -euo pipefail
base=/work/gm64/m64000/zipa/mfa_miyabi_20260922
export MFA_PYTHON=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python
export PYTHONPATH="$base/python_site:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}
export MFA_TMPDIR=/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NCCL_DEBUG=WARN NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none
export WANDB_MODE=online PYTHONUNBUFFERED=1
source "$base/zipa/experiments/mfa_zapformer/env.sh"
export RANK=${OMPI_COMM_WORLD_RANK:?} WORLD_SIZE=8 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
export MASTER_ADDR=${COST_MASTER_ADDR:?} MASTER_PORT=${COST_MASTER_PORT:?}
export CUDA_VISIBLE_DEVICES=0
master_ip=$(getent ahostsv4 "$MASTER_ADDR" | awk 'NR==1{print $1}')
peer_ip=$master_ip
if [[ "$RANK" == 0 ]]; then
  peer_host=$(sed -n '2p' "$COST_OUTPUT/nodes")
  peer_ip=$(getent ahostsv4 "$peer_host" | awk 'NR==1{print $1}')
fi
interface=$(ip route get "$peer_ip" | awk '{for(i=1;i<=NF;i++)if($i=="dev"){print $(i+1);exit}}')
test -n "$interface"
test "$interface" != lo
export MASTER_ADDR=$master_ip
export GLOO_SOCKET_IFNAME=$interface
printf 'NETWORK rank=%s master=%s peer=%s interface=%s\n' "$RANK" "$MASTER_ADDR" "$peer_ip" "$interface"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv -l 1 > "$COST_OUTPUT/gpu_rank${RANK}.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
if [[ "$RANK" == 0 ]]; then
  cp /work/gm64/m64000/OTTC/reproduction/english_suite/summarize_gpu_cost.py "$COST_OUTPUT/source/"
fi
if "$MFA_PYTHON" -u "$COST_OUTPUT/source/benchmark_mfa_gpu_cost.py" --output "$COST_OUTPUT"; then
  result=0
else
  result=$?
fi
if [[ "$RANK" == 0 ]]; then
  "$MFA_PYTHON" "$COST_OUTPUT/source/summarize_gpu_cost.py" --mfa-dir "$COST_OUTPUT" --output "$COST_OUTPUT/summary.md"
  "$MFA_PYTHON" "$COST_OUTPUT/source/summarize_gpu_cost.py" --mfa-dir "$COST_OUTPUT"
fi
exit "$result"
