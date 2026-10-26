#!/bin/bash
set -euo pipefail
rank=${OMPI_COMM_WORLD_RANK:?}
costs=(ce jsd_standard ag jdiv)
seeds=(42 43 44)
name="${costs[$((rank / 3))]}__seed${seeds[$((rank % 3))]}"
stage=${1:?}
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
root="$PWD/reproduction/buckeye_full40_retry48_20261026"
evidence="$root/jobs/${PBS_JOBID:?}/$name"
mkdir -p "$evidence"
exec >> "$evidence/$stage.log" 2>&1
export CUDA_VISIBLE_DEVICES=0 OTTC_EVAL_UPLOAD=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    printf '{"recipe":"%s","stage":"%s","exit_code":%s}\n' "$name" "$stage" "$exit_code" > "$evidence/${stage}_EXIT.json"
    # A clean checkpointed pause is handled by the allocation controller.
    if [[ $exit_code == 75 ]]; then exit 0; fi
    exit "$exit_code"
}
trap finish EXIT
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv -l 10 >> "$evidence/${stage}_telemetry.csv" &
monitor=$!
if [[ $stage == gate ]]; then
    "$LIBRI_PYTHON" -u -m reproduction.buckeye_full40_retry48_20261026.runtime --recipe "$name" --stage gate
else
    remaining=$(( ${BUCKEYE_DEADLINE:?} - $(date +%s) - 5400 ))
    (( remaining > 300 ))
    "$LIBRI_PYTHON" -u -m reproduction.buckeye_full40_retry48_20261026.runtime --recipe "$name" --stage train --max-seconds "$remaining"
    "$LIBRI_PYTHON" -u -m reproduction.buckeye_full40_seed_sweep_20261025.evaluate --recipe "$name"
fi
