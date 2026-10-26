#!/bin/bash
set -euo pipefail
rank=${OMPI_COMM_WORLD_RANK:?}
costs=(ce jsd_standard ag jdiv)
seeds=(42 43 44)
name="${costs[$((rank / 3))]}__seed${seeds[$((rank % 3))]}"
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
root="$PWD/reproduction/buckeye_full40_exactfa_20261026"
evidence="$root/jobs/${PBS_JOBID:?}/$name"
mkdir -p "$evidence"
exec >> "$evidence/evaluate.log" 2>&1
export CUDA_VISIBLE_DEVICES=0 OTTC_EVAL_UPLOAD=0 WANDB_MODE=disabled
export NUMBA_CACHE_DIR="$root/numba_cache/$name"
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK
while IFS= read -r variable; do
    case "$variable" in
        OMPI_COMM_WORLD_*|PMI_*|PMIX_*|MPI_LOCALRANKID|MPI_LOCALNRANKS) unset "$variable" ;;
    esac
done < <(compgen -e)
finish() { status=$?; printf '{"recipe":"%s","exit_code":%s}\n' "$name" "$status" > "$evidence/EXIT.json"; }
trap finish EXIT
"$LIBRI_PYTHON" -u -m reproduction.buckeye_full40_exactfa_20261026.evaluate --recipe "$name"
