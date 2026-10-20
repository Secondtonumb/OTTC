#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
task_dir=/work/gm64/m64000/OTTC/reproduction/zapformer_s3t2ss_gpu_profile_20261012
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
export NATIVE_PROFILE_DURATION="${NATIVE_PROFILE_DURATION:?Set duration from successful single-GPU measurements}"
export NATIVE_PROFILE_OUTPUT="$task_dir/ddp_${NATIVE_PROFILE_NODES}_${NATIVE_PROFILE_DURATION}_${PBS_JOBID%%.*}"
task_allocation="$task_dir/allocation_${PBS_JOBID%%.*}"
mkdir -p "$task_allocation"
awk '!seen[$0]++' "${PBS_NODEFILE:?}" > "$task_allocation/nodes"
IFS= read -r NATIVE_PROFILE_MASTER < "$task_allocation/nodes"
export NATIVE_PROFILE_MASTER NATIVE_PROFILE_PORT=$((29500 + ${PBS_JOBID%%.*} % 1000))
export OMPI_MCA_mca_base_env_list='NATIVE_PROFILE_NODES;NATIVE_PROFILE_DURATION;NATIVE_PROFILE_OUTPUT;NATIVE_PROFILE_MASTER;NATIVE_PROFILE_PORT'
mpirun -np "$NATIVE_PROFILE_NODES" --hostfile "$task_allocation/nodes" --map-by ppr:1:node --bind-to none \
  bash "$task_dir/profile_8gpu_worker.sh"
"$MFA_PYTHON" "$task_dir/summarize.py" "$task_dir"
