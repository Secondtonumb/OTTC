#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
task_stage=/work/gm64/m64000/OTTC/reproduction/zapformer_libri100_ablation_20261013
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
task_arm="${SPOTTC_ARM:?Set SPOTTC_ARM to jsd_only, ce_fs or ce_only}"
"$MFA_PYTHON" "$task_stage/verify.py" --arm "$task_arm"
export JSD50_OUTPUT="$task_stage/models/$task_arm"
exec 9> "$JSD50_OUTPUT/.training.lock"
flock -n 9 || { echo 'Continuation already running'; exit 1; }
if [[ -s "$JSD50_OUTPUT/TRAIN_COMPLETE.json" ]]; then exit 0; fi
awk '!seen[$0]++' "${PBS_NODEFILE:?}" > "$JSD50_OUTPUT/nodes"
test "$(wc -l < "$JSD50_OUTPUT/nodes")" -eq 8
IFS= read -r JSD50_MASTER < "$JSD50_OUTPUT/nodes"
export JSD50_MASTER JSD50_PORT=$((29500 + ${PBS_JOBID%%.*} % 1000))
export WANDB_MODE="${WANDB_MODE:-online}"
export OMPI_MCA_mca_base_env_list='JSD50_MASTER;JSD50_PORT;JSD50_OUTPUT;JSD50_WALL_SECONDS;WANDB_MODE;WANDB_PROJECT;WANDB_ENTITY'
mpirun -np 8 --hostfile "$JSD50_OUTPUT/nodes" --map-by ppr:1:node --bind-to none \
  bash "$task_stage/worker.sh"
