#!/bin/bash
set -euo pipefail
rank=${OMPI_COMM_WORLD_RANK:?MPI dispatch rank required}
cost=${FRAME_DIV_COST:?}
[[ $rank =~ ^[0-2]$ ]]
case $cost in jsd_standard|jdiv|ag) ;; *) exit 2 ;; esac
seed=$((42 + rank))
name="frame_${cost}__seed${seed}"
# Remove dispatch-only rank information before HF Accelerator inspects the environment.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK
while IFS= read -r variable; do
    case "$variable" in
        OMPI_COMM_WORLD_*|PMI_*|PMIX_*|MPI_LOCALRANKID|MPI_LOCALNRANKS) unset "$variable" ;;
    esac
done < <(compgen -e)
export CUDA_VISIBLE_DEVICES=0 OTTC_EVAL_UPLOAD=0
if [[ ${1:-} == --dry-run ]]; then
    printf 'rank=%s recipe=%s GPUs=1 batch=64 epochs=40 world_size=1 independent=true\n' "$rank" "$name"
    python3 - <<'PY'
import os
assert not any(k in os.environ for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'))
assert not any(k.startswith(('OMPI_COMM_WORLD_', 'PMI_', 'PMIX_')) for k in os.environ)
PY
    exit 0
fi
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
root="$PWD/reproduction/timit_frame_divergences_20261026"
evidence="$root/jobs/${PBS_JOBID:?}/$name"
mkdir -p "$evidence"
exec >> "$evidence/worker.log" 2>&1
export WANDB_CACHE_DIR="$evidence/wandb_cache" WANDB_DATA_DIR="$evidence/wandb_data"
export WANDB_RUN_GROUP=timit_frame_divergences_20261026
monitor=''
finish() {
    status=$?
    if [[ -n $monitor ]]; then kill "$monitor" 2>/dev/null || true; fi
    printf '{"recipe":"%s","rank":%s,"seed":%s,"exit_code":%s,"finished_utc":"%s"}\n' \
        "$name" "$rank" "$seed" "$status" "$(date -u +%FT%TZ)" > "$evidence/WORKER_EXIT.json.tmp"
    mv "$evidence/WORKER_EXIT.json.tmp" "$evidence/WORKER_EXIT.json"
    # Other independent seeds finish; the head verifies all true exit codes.
    exit 0
}
trap finish EXIT
hostname > "$evidence/host.txt"
printf 'job=%s\nrank=%s\nrecipe=%s\nCUDA_VISIBLE_DEVICES=%s\nworld_size=1\n' \
    "$PBS_JOBID" "$rank" "$name" "$CUDA_VISIBLE_DEVICES" > "$evidence/runtime_environment.txt"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 5 \
    >> "$evidence/telemetry.csv" &
monitor=$!
"$LIBRI_PYTHON" -u -m reproduction.timit_frame_divergences_20261026.train --recipe "$name" --profile
remaining=$(( ${FRAME_DIV_DEADLINE:?} - $(date +%s) - 1200 ))
(( remaining > 300 ))
"$LIBRI_PYTHON" -u -m reproduction.timit_frame_divergences_20261026.train --recipe "$name" --max-seconds "$remaining"
