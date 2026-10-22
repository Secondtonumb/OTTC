#!/usr/bin/env bash
# Execute in an existing one-GPU allocation; never submits a job.
set -euo pipefail
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$recipe_dir/../.." && pwd)"
read_only=false
for arg in "$@"; do
    case "$arg" in --help|-h|--dry-run) read_only=true ;; esac
done
if ! "$read_only" && [[ -z "${PBS_JOBID:-}" ]]; then
    echo '请在已有的 interactive GPU 节点运行。只读检查可加 --dry-run。' >&2
    exit 2
fi
cd -- "$repo_root"
training_python="${OTTC_PYTHON:-/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python}"
export PYTHONPATH="$repo_root/reproduction/vendor:$repo_root/reproduction:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$(dirname -- "$(dirname -- "$training_python")")/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OTTC_EVAL_UPLOAD=0
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$repo_root/.cache/wandb/cache}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$repo_root/.cache/wandb/data}"
exec "$training_python" -u -m reproduction.timit_divergences_20261022.train "$@"
