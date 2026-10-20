#!/usr/bin/env bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
case "${OMPI_COMM_WORLD_RANK:-0}" in
  0) objective=ctc;;
  1) objective=ce;;
  2) objective=jsd_upstream;;
  *) exit 2;;
esac
# MPI launches independent configurations. Prevent Accelerate from treating
# these processes as workers of one distributed training run.
for launch_env_name in $(compgen -e); do
  case "$launch_env_name" in
    OMPI_*|PMI_*|PMIX_*|MV2_*|MPI_LOCAL*|RANK|WORLD_SIZE|LOCAL_RANK|LOCAL_WORLD_SIZE|GROUP_RANK|ROLE_RANK|ROLE_WORLD_SIZE|MASTER_ADDR|MASTER_PORT|ACCELERATE_*)
      unset "$launch_env_name";;
  esac
done
export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_GROUP=timit_fourphonesets_seed42
if [[ "${FOURHEAD_LAUNCH_CHECK_ONLY:-0}" == 1 ]]; then
  /work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python -c 'from accelerate import PartialState; s=PartialState(cpu=True); assert s.num_processes == 1; print("Independent launch: world_size=1")'
  exit
fi
out="reproduction/multihead_timit/models/${objective}"
mkdir -p "$out"
# Once an objective has its own PBS job, the old interactive loop must skip it.
if [[ -f "reproduction/multihead_timit/parallel_dispatch/${objective}.job" &&
      "${FOURHEAD_EXTERNAL_JOB:-0}" != 1 ]]; then
  printf '%s dispatched to independent PBS job %s\n' "$objective" "$(cat "reproduction/multihead_timit/parallel_dispatch/${objective}.job")"
  exit
fi
# Interactive fallback and batch resubmission may reach the same objective.
# Serialize writers and skip completed experiments without changing run IDs.
exec 9> "$out/.training.lock"
flock 9
if [[ -f "$out/TRAIN_COMPLETE.json" ]]; then
  printf '%s already complete\n' "$objective"
  exit
fi
runtime_arguments=()
if [[ "${FOURHEAD_MAX_RUNTIME_SECONDS:-0}" != 0 ]]; then
  runtime_arguments=(--max-runtime-seconds "$FOURHEAD_MAX_RUNTIME_SECONDS")
fi
/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python -u reproduction/train_multihead.py \
  --data reproduction/multihead_timit/data --objective "$objective" --output "$out" \
  --epochs 40 --freeze-alignment-epochs 30 --batch-size 16 --effective-batch-size 64 \
  "${runtime_arguments[@]}" \
  >> "$out/train.log" 2>&1
