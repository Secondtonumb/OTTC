#!/usr/bin/env bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
case "${OMPI_COMM_WORLD_RANK:?}" in
  0) objective=ctc;;
  1) objective=ce;;
  2) objective=jsd_upstream;;
  *) exit 2;;
esac
for launch_env_name in $(compgen -e); do
  case "$launch_env_name" in
    OMPI_*|PMI_*|PMIX_*|MV2_*|MPI_LOCAL*|RANK|WORLD_SIZE|LOCAL_RANK|LOCAL_WORLD_SIZE|GROUP_RANK|ROLE_RANK|ROLE_WORLD_SIZE|MASTER_ADDR|MASTER_PORT|ACCELERATE_*) unset "$launch_env_name";;
  esac
done
export CUDA_VISIBLE_DEVICES=0
task_python=/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python
for dataset in libri100 libri960; do
  output="reproduction/multihead_${dataset}/models/${objective}"
  mkdir -p "$output"
  exec 9> "$output/.training.lock"
  flock 9
  if [[ -f "$output/TRAIN_COMPLETE.json" ]]; then
    flock -u 9
    continue
  fi
  seconds_remaining=$((ENGLISH_SUITE_DEADLINE-$(date +%s)))
  if (( seconds_remaining < 3600 )); then exit 0; fi
  export WANDB_RUN_GROUP="${dataset}_fourphonesets_seed42"
  "$task_python" -u reproduction/train_multihead.py \
    --data "reproduction/multihead_${dataset}/data" --objective "$objective" --output "$output" \
    --epochs 40 --freeze-alignment-epochs 30 --batch-size 4 --effective-batch-size 64 \
    --max-duration 0 --max-runtime-seconds "$seconds_remaining" >> "$output/train.log" 2>&1
  flock -u 9
  if [[ ! -f "$output/TRAIN_COMPLETE.json" ]]; then exit 0; fi
done
