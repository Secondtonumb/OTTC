#!/usr/bin/env bash
# Run inside a Miyabi PBS GPU interactive allocation. Evaluation/probe only.
set -euo pipefail
probe_recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
probe_repo_root="$(cd -- "$probe_recipe_dir/../.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
先申请 Miyabi GPU 交互节点：
  qsub -I -q interact-g -W group_list=gm64 -l select=1:ncpus=16:mpiprocs=1 -l walltime=00:30:00
进入节点后：
  cd /home/m64000/work/OTTC
  bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh

默认：原始SSL初始化的 native10ms mono/upstream-JSD，真实TIMIT train的2条语音，1次优化器更新。
可选参数：
  --weight 0.1 --tau 4 --steps 1 --batch-size 2
  --cost jsd_upstream|ce|jsd_standard --head mono|s3t2ss
  --core-only                 只测试底层CUDA loss，不加载编码器或语音
  --output-dir NEW_DIRECTORY  指定新的本地输出目录
全部结果本地保存；不上传W&B，不覆盖旧模型或resume。
USAGE
    exit 0
fi

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo '请先运行 qsub -I -q interact-g -W group_list=gm64 -l select=1:ncpus=16:mpiprocs=1 -l walltime=00:30:00' >&2
    echo '进入已分配的GPU节点后，再运行本脚本；脚本不会自动提交训练。' >&2
    exit 2
fi

cd -- "$probe_repo_root"
probe_python="${OTTC_PYTHON:-/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python}"
export PYTHONPATH="$probe_repo_root/reproduction/vendor:$probe_repo_root${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled OTTC_EVAL_UPLOAD=0 TOKENIZERS_PARALLELISM=false
exec "$probe_python" -u "$probe_recipe_dir/gpu_probe.py" "$@"
