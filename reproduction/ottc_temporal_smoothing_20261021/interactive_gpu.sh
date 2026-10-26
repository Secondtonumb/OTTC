#!/usr/bin/env bash
# Full training inside an existing Miyabi PBS GPU allocation; --probe retains smoke checks.
set -euo pipefail
probe_recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
probe_repo_root="$(cd -- "$probe_recipe_dir/../.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'USAGE'
在已有 Miyabi PBS GPU 交互节点运行（请申请足够完成训练的 walltime）：
  cd /work/gm64/m64000/OTTC
  bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh

默认正式训练：完整 TIMIT train，40 epochs，每次更新 batch size 64，无 steps 上限。
原始 SSL 初始化、native10ms、mono、upstream-JSD + temporal smoothing。
参数：
  --epochs 40 --batch-size 64 --weight 0.1 --tau 4
  --cost jsd_upstream|ce|jsd_standard --head mono|s3t2ss
  --output-dir NEW_DIRECTORY       新训练目录（默认自动生成）
  --resume --output-dir DIRECTORY  续训同一配置
  --profile                       仅运行精确配置 GPU profile
  --max-seconds N                  可选主动暂停预算；默认不限制
训练前自动 profile，按显存/吞吐选择 microbatch，完整 optimizer batch 仍为64。
按 dev PER 保留最佳3份权重，另保留最新完整续训状态；不做final5/final10平均。
训练标量记录 W&B；profile、独立评估及预测只保存在本地。
短测试（保留旧行为）：
  bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --probe --steps 1 --batch-size 2
  bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --core-only
USAGE
    exit 0
fi

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo '请先通过 qsub -I -q interact-g -W group_list=gm64 申请GPU交互节点，并设置足够完成训练的walltime。' >&2
    echo '进入已分配的GPU节点后，再运行本脚本；脚本不会自动提交训练。' >&2
    exit 2
fi

cd -- "$probe_repo_root"
probe_python="${OTTC_PYTHON:-/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python}"
export PYTHONPATH="$probe_repo_root/reproduction/vendor:$probe_repo_root${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export OTTC_EVAL_UPLOAD=0 TOKENIZERS_PARALLELISM=false
probe_mode=false
probe_arguments=()
for probe_arg in "$@"; do
    case "$probe_arg" in
        --probe) probe_mode=true ;;
        --core-only) probe_mode=true; probe_arguments+=("$probe_arg") ;;
        *) probe_arguments+=("$probe_arg") ;;
    esac
done
if "$probe_mode"; then
    export WANDB_MODE=disabled
    exec "$probe_python" -u "$probe_recipe_dir/gpu_probe.py" "${probe_arguments[@]}"
fi
exec "$probe_python" -u -m reproduction.ottc_temporal_smoothing_20261021.train "${probe_arguments[@]}"
