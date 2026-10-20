#!/bin/bash
set -euo pipefail
base=/work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016
test -n "${PBS_JOBID:-}"
task_file=${1:?Task JSON required}
allocation_nodes=${2:?Allocated node count required}
printf 'INTERACTIVE_STAGE_START %s %s pool\n' "$PBS_JOBID" "$(hostname)"
if [[ "$allocation_nodes" == 1 ]]; then
    /work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python "$base/interactive_rank.py" "$task_file" --one-node
else
    mpirun -np "$allocation_nodes" --hostfile "${PBS_NODEFILE:?}" --map-by ppr:1:node --bind-to none \
        /work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python "$base/interactive_rank.py" "$task_file"
fi
printf 'INTERACTIVE_STAGE_END %s pool\n' "$PBS_JOBID"
