#!/bin/bash
set -euo pipefail
rank=${OMPI_COMM_WORLD_RANK:?}
costs=(ce jsd_standard ag jdiv)
seeds=(42 43 44)
(( rank >= 0 && rank < 12 ))
name="${costs[$((rank / 3))]}__seed${seeds[$((rank % 3))]}"
if [[ ${1:-} == --dry-run ]]; then
    echo "rank=$rank recipe=$name physical_batch=64 world_size=1"
    exit 0
fi
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
root="$PWD/reproduction/buckeye_full40_seed_sweep_20261025"
evidence="$root/jobs/${PBS_JOBID:?}/$name"
mkdir -p "$evidence"
exec >> "$evidence/worker.log" 2>&1
export CUDA_VISIBLE_DEVICES=0 OTTC_EVAL_UPLOAD=0
export WANDB_CACHE_DIR="$evidence/wandb_cache" WANDB_DATA_DIR="$evidence/wandb_data"
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK
while IFS= read -r variable; do
    case "$variable" in
        OMPI_COMM_WORLD_*|PMI_*|PMIX_*|MPI_LOCALRANKID|MPI_LOCALNRANKS) unset "$variable" ;;
    esac
done < <(compgen -e)
monitor=''
finish() {
    exit_code=$?
    if [[ -n $monitor ]]; then kill "$monitor" 2>/dev/null || true; fi
    printf '{"recipe":"%s","exit_code":%s}\n' "$name" "$exit_code" > "$evidence/WORKER_EXIT.json"
    exit 0
}
trap finish EXIT
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 10 >> "$evidence/telemetry.csv" &
monitor=$!
if [[ ${1:-} == --smoke ]]; then
    if [[ ! -f "$root/profiles/$name/PASS.json" ]]; then
        "$LIBRI_PYTHON" -u -m reproduction.buckeye_full40_seed_sweep_20261025.train --recipe "$name" --profile
    fi
    "$LIBRI_PYTHON" -u -m reproduction.buckeye_full40_seed_sweep_20261025.smoke --recipe "$name"
    exit 0
fi
remaining=$(( ${BUCKEYE_DEADLINE:?} - $(date +%s) - 7200 ))
(( remaining > 300 ))
"$LIBRI_PYTHON" -u -m reproduction.buckeye_full40_seed_sweep_20261025.train --recipe "$name" --max-seconds "$remaining"
"$LIBRI_PYTHON" -u -m reproduction.buckeye_full40_seed_sweep_20261025.evaluate --recipe "$name"
"$LIBRI_PYTHON" -u -m reproduction.buckeye_full40_seed_sweep_20261025.report