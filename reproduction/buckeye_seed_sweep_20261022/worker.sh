#!/bin/bash
set -euo pipefail
rank=${OMPI_COMM_WORLD_RANK:?MPI worker rank required}
costs=(ce jsd_standard ag jdiv)
seeds=(42 43 44)
(( rank >= 0 && rank < 12 ))
cost=${costs[$((rank / 3))]}
seed=${seeds[$((rank % 3))]}
name="${cost}__seed${seed}"
if [[ ${1:-} == --dry-run ]]; then
    echo "rank=$rank recipe=$name GPUs=1 batch=64 epochs=40 world_size=1 independent=true"
    exit 0
fi
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
root="$PWD/reproduction/buckeye_seed_sweep_20261022"
evidence="$root/jobs/${PBS_JOBID:?}/$name"
mkdir -p "$evidence"
exec >> "$evidence/worker.log" 2>&1
export CUDA_VISIBLE_DEVICES=0 OTTC_EVAL_UPLOAD=0
export WANDB_CACHE_DIR="$evidence/wandb_cache" WANDB_DATA_DIR="$evidence/wandb_data"
export WANDB_RUN_GROUP=buckeye_seed_sweep_20261022
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK
while IFS= read -r variable; do
    case "$variable" in
        OMPI_COMM_WORLD_*|PMI_*|PMIX_*|MPI_LOCALRANKID|MPI_LOCALNRANKS) unset "$variable" ;;
    esac
done < <(compgen -e)
monitor=''
finish() {
    status=$?
    if [[ -n $monitor ]]; then kill "$monitor" 2>/dev/null || true; fi
    printf '{"recipe":"%s","rank":%s,"exit_code":%s,"finished_utc":"%s"}\n' \
        "$name" "$rank" "$status" "$(date -u +%FT%TZ)" > "$evidence/WORKER_EXIT.json"
    exit 0
}
trap finish EXIT
hostname > "$evidence/host.txt"
env | sort > "$evidence/runtime_environment.txt"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 10 \
    >> "$evidence/telemetry.csv" &
monitor=$!
if [[ ! -f "$root/profiles/$name/PASS.json" ]]; then
    "$LIBRI_PYTHON" -u -m reproduction.buckeye_seed_sweep_20261022.train --recipe "$name" --profile
fi
remaining=$(( ${BUCKEYE_DEADLINE:?} - $(date +%s) - 1800 ))
(( remaining > 300 ))
"$LIBRI_PYTHON" -u -m reproduction.buckeye_seed_sweep_20261022.train --recipe "$name" --max-seconds "$remaining"
"$LIBRI_PYTHON" -u -m reproduction.buckeye_seed_sweep_20261022.report
