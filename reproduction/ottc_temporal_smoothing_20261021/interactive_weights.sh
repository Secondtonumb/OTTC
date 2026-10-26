#!/usr/bin/env bash
# Sequential weight comparison inside an existing PBS GPU allocation.
set -euo pipefail
smooth_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
    cat <<'USAGE'
在已有 PBS GPU 交互节点运行：
  bash reproduction/ottc_temporal_smoothing_20261021/interactive_weights.sh
依次训练 weight=0.01、0.001，均为40 epochs、batch64、tau=4。
每个实验自动profile，各保留dev最好的3份权重及最新完整续训状态。
也可只跑其中一个：
  bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --weight 0.01
  bash reproduction/ottc_temporal_smoothing_20261021/interactive_gpu.sh --weight 0.001
可选：设置 SMOOTH_SWEEP_DIR 为专用比较目录；重新使用同一目录时自动续训。
此脚本独立启动交互实验，不会取消已经提交的PBS实验。
USAGE
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo '此入口不接收参数；自定义单个实验请使用 interactive_gpu.sh。' >&2
    exit 2
fi
if [[ -z "${PBS_JOBID:-}" ]]; then
    echo '请先申请PBS GPU交互节点，再运行此脚本。' >&2
    exit 2
fi
smooth_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
smooth_root="${SMOOTH_SWEEP_DIR:-$smooth_dir/training/interactive_weights_${smooth_stamp}_$$}"
mkdir -p "$smooth_root"
smooth_root="$(cd -- "$smooth_root" && pwd)"
for smooth_weight in 0.01 0.001; do
    smooth_output="$smooth_root/weight_$smooth_weight"
    smooth_resume=()
    if [[ -f "$smooth_output/run_config.json" ]]; then
        smooth_resume+=(--resume)
    fi
    echo "Training weight=$smooth_weight epochs=40 batch=64 tau=4 output=$smooth_output"
    bash "$smooth_dir/interactive_gpu.sh" \
        --weight "$smooth_weight" --tau 4 --epochs 40 --batch-size 64 \
        --output-dir "$smooth_output" "${smooth_resume[@]}" \
        2>&1 | tee -a "$smooth_root/weight_$smooth_weight.log"
done
