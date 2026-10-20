#!/bin/bash
set -euo pipefail
cd /work/gm64/m64000/OTTC
phase=${1:?phase}
case "${OMPI_COMM_WORLD_RANK:?}" in
    0) objective=ce;;
    1) objective=jsd_upstream;;
    2) objective=jsd_standard;;
    *) exit 2;;
esac
# MPI assigns independent experiments; these are single-process Trainers.
for launch_env_name in $(compgen -e); do
    case "$launch_env_name" in
        OMPI_*|PMI_*|PMIX_*|MV2_*|MPI_LOCAL*|RANK|WORLD_SIZE|LOCAL_RANK|LOCAL_WORLD_SIZE|GROUP_RANK|ROLE_RANK|ROLE_WORLD_SIZE|MASTER_ADDR|MASTER_PORT|ACCELERATE_*) unset "$launch_env_name";;
    esac
done
status=0
bash reproduction/libri100_singlehead_20261014/worker.sh "$objective" "$phase" 0 \
    >> "reproduction/libri100_singlehead_20261014/${phase}_${objective}.log" 2>&1 || status=$?
printf '%s\n' "$status" > "reproduction/libri100_singlehead_20261014/${phase}_${objective}_exit_code.txt"
# Allow every independently paused Trainer to finish its optimizer save.
if [[ "$status" == 75 ]]; then exit 0; fi
exit "$status"
