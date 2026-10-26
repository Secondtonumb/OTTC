#!/bin/bash
set -euo pipefail
rank=${OMPI_COMM_WORLD_RANK:?MPI dispatch rank required}
cost=${BUCKEYE_FRAME_COST:?}
[[ $rank =~ ^[0-2]$ ]]
case $cost in ce|jsd_standard|ag|jdiv) ;; *) exit 2 ;; esac
seed=$((42 + rank))
name="frame_${cost}__seed${seed}"
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
assert not any(k in os.environ for k in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'))
assert not any(k.startswith(('OMPI_COMM_WORLD_','PMI_','PMIX_')) for k in os.environ)
PY
    exit 0
fi
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
root="$PWD/reproduction/buckeye_frame_sweep_20261027"
evidence="$root/jobs/${PBS_JOBID:?}/$name"
mkdir -p "$evidence"
exec >> "$evidence/worker.log" 2>&1
export WANDB_CACHE_DIR="$evidence/wandb_cache" WANDB_DATA_DIR="$evidence/wandb_data"
export WANDB_RUN_GROUP=buckeye_frame_sweep_20261027
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
monitor=''
finish() {
    worker_status=$?
    if [[ -n $monitor ]]; then kill "$monitor" 2>/dev/null || true; fi
    printf '{"recipe":"%s","rank":%s,"seed":%s,"exit_code":%s,"finished_utc":"%s"}\n' \
        "$name" "$rank" "$seed" "$worker_status" "$(date -u +%FT%TZ)" > "$evidence/WORKER_EXIT.json.tmp"
    mv "$evidence/WORKER_EXIT.json.tmp" "$evidence/WORKER_EXIT.json"
    # MPI waits for every independent seed; the allocation checks actual exits.
    exit 0
}
trap finish EXIT
hostname > "$evidence/host.txt"
printf 'job=%s\nrank=%s\nrecipe=%s\nCUDA_VISIBLE_DEVICES=%s\nworld_size=1\n' \
    "$PBS_JOBID" "$rank" "$name" "$CUDA_VISIBLE_DEVICES" > "$evidence/runtime_environment.txt"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 10 \
    >> "$evidence/telemetry.csv" &
monitor=$!
"$LIBRI_PYTHON" -u -m reproduction.buckeye_frame_sweep_20261027.train --recipe "$name" --profile
remaining=$(( ${BUCKEYE_FRAME_DEADLINE:?} - $(date +%s) - 3600 ))
(( remaining > 300 ))
"$LIBRI_PYTHON" -u -m reproduction.buckeye_frame_sweep_20261027.train --recipe "$name" --max-seconds "$remaining"
"$LIBRI_PYTHON" -B -m reproduction.buckeye_frame_sweep_20261027.report
