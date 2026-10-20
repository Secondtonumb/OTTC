#!/bin/bash
set -euo pipefail
task_dir=/work/gm64/m64000/OTTC/reproduction/zapformer_s3t2_native_20261012
cd /work/gm64/m64000/OTTC
source "$task_dir/environment.sh"
"$MFA_PYTHON" "$task_dir/validate.py" --check-acceptance all
task_output="$task_dir/training_${S3T2_STAGE:?}"
mkdir -p "$task_output"
exec 9> "$task_output/.training.lock"
flock -n 9 || { echo 'This experiment is already running'; exit 1; }
if [[ -s "$task_output/TRAIN_COMPLETE.json" ]]; then exit 0; fi
awk '!seen[$0]++' "${PBS_NODEFILE:?}" > "$task_output/nodes"
IFS= read -r S3T2_MASTER_ADDR < "$task_output/nodes"
export S3T2_MASTER_ADDR
export S3T2_MASTER_PORT=$((29500 + ${PBS_JOBID%%.*} % 1000))
export OMPI_MCA_mca_base_env_list='S3T2_STAGE;S3T2_NODES;S3T2_EPOCHS;S3T2_LR;S3T2_MAX_DURATION;S3T2_INIT_CHECKPOINT;S3T2_MASTER_ADDR;S3T2_MASTER_PORT;WANDB_MODE'
mpirun -np "$S3T2_NODES" --hostfile "$task_output/nodes" --map-by ppr:1:node --bind-to none bash "$task_dir/worker.sh"
# Resume explicitly with the same PBS script after a normal wall-time checkpoint.
# A failed process exits above; it never automatically queues another failure.
