#!/bin/bash
set -euo pipefail
base=/work/gm64/m64000/OTTC/reproduction/libri100_singlehead_average_20261016
task=${1:?Specify a short task}
test -n "${PBS_JOBID:-}"
printf 'INTERACTIVE_STAGE_START %s %s %s\n' "$PBS_JOBID" "$(hostname)" "$task"
pids=()
for objective in jsd_upstream ce jsd_standard; do
    if [[ -f "$base/short_tasks/$objective/${task}_COMPLETE.json" ]]; then continue; fi
    LIBRI_OBJECTIVE=$objective LIBRI_AVG_TASK=$task bash "$base/run_one.pbs" &
    pids+=("$!")
done
stage_failed=0
for stage_pid in "${pids[@]}"; do
    if ! wait "$stage_pid"; then stage_failed=1; fi
done
printf 'INTERACTIVE_STAGE_END %s %s %s\n' "$PBS_JOBID" "$task" "$stage_failed"
exit "$stage_failed"
