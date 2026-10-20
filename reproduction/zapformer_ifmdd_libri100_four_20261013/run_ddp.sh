#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
task_stage=/work/gm64/m64000/OTTC/reproduction/zapformer_ifmdd_libri100_four_20261013
"$MFA_PYTHON" "$task_stage/verify.py" --arm "${SPOTTC_ARM:?}"
test -s "$task_stage/ALL_ARMS_PASS.json"
export SP4_OUTPUT="$task_stage/models/$SPOTTC_ARM"
mkdir -p "$SP4_OUTPUT"
exec 9> "$SP4_OUTPUT/.training.lock"
flock -n 9 || { echo 'Same arm is already running'; exit 1; }
if [[ -s "$SP4_OUTPUT/TRAIN_COMPLETE.json" ]]; then exit 0; fi
awk '!seen[$0]++' "${PBS_NODEFILE:?}" > "$SP4_OUTPUT/nodes"
test "$(wc -l < "$SP4_OUTPUT/nodes")" -eq 8
IFS= read -r SP4_MASTER < "$SP4_OUTPUT/nodes"
export SP4_MASTER SP4_PORT=$((29500 + ${PBS_JOBID%%.*} % 1000))
export WANDB_MODE="${WANDB_MODE:-online}"
export OMPI_MCA_mca_base_env_list='SP4_MASTER;SP4_PORT;SP4_OUTPUT;SP4_WALL_SECONDS;SPOTTC_ARM;WANDB_MODE;WANDB_PROJECT;WANDB_ENTITY'
mpirun -np 8 --hostfile "$SP4_OUTPUT/nodes" --map-by ppr:1:node --bind-to none bash "$task_stage/worker.sh"
