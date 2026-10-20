#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
task_stage=/work/gm64/m64000/OTTC/reproduction/zapformer_libri100_s3t2ss_native_20261012
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
"$MFA_PYTHON" "$task_stage/verify_8gpu.py"
task_output="$task_stage/training"
mkdir -p "$task_output"
exec 9> "$task_output/.training.lock"
flock -n 9 || { echo 'Experiment already running'; exit 1; }
if [[ -s "$task_output/TRAIN_COMPLETE.json" ]]; then exit 0; fi
awk '!seen[$0]++' "${PBS_NODEFILE:?}" > "$task_output/nodes"
test "$(wc -l < "$task_output/nodes")" -eq 8
IFS= read -r NATIVE_TRAIN_MASTER < "$task_output/nodes"
export NATIVE_TRAIN_MASTER NATIVE_TRAIN_PORT=$((29500 + ${PBS_JOBID%%.*} % 1000))
export WANDB_MODE="${WANDB_MODE:-online}"
export OMPI_MCA_mca_base_env_list='NATIVE_TRAIN_MASTER;NATIVE_TRAIN_PORT;NATIVE_TASK_WALL_SECONDS;WANDB_MODE;WANDB_PROJECT;WANDB_ENTITY'
mpirun -np 8 --hostfile "$task_output/nodes" --map-by ppr:1:node --bind-to none \
  bash "$task_stage/ddp_worker.sh"
