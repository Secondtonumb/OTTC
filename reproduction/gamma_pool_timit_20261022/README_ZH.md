# OTTC gamma 加权位置解码与 TIMIT 对照

核心代码：`ottc/utils/gamma_decode.py`。这是推理端改动，复用已训练权重。

```python
from ottc.utils.gamma_decode import gamma_decode, pool_gamma_logits

# logits [T,K], alpha [T]，只包含有效帧；M 必须由调用方给出。
token_ids = gamma_decode(logits, alpha, num_tokens=M)
# 加权概率平均（不同的推理规则）：
token_ids_prob = gamma_decode(logits, alpha, num_tokens=M, aggregation='posterior')

# 如果已经保存了 gamma [T,M]：
token_ids = pool_gamma_logits(logits, gamma).argmax(-1)
# 几何聚合时 logits 可以换成 log posterior。概率平均需要 posterior 值。
```

alpha 的累积质量区间与均匀 beta=1/M 的区间求交得到 gamma。跨边界帧可向多个位置分摊质量。
每列归一化后计算 `(gamma / gamma.sum(0)).T @ logits`，每个位置输出一次 argmax。
输出位置不做 CTC 重复合并；相邻同音素位置保留。无质量帧和 padding 不参与聚合。
提供的 gamma 也支持非均匀 beta。对固定 gamma，这是 CE 训练代价的逐位置最小化；
对 upstream-JSD 权重则是单独测试的推理规则，不能称为其原始 JSD 目标的精确最优决策。

通用评测入口 `run_exp.py` / `build_compute_metrics` 新增可选参数。
`--ottc_aggregation logits`（默认）使用几何规则，`--ottc_aggregation posterior` 使用加权概率平均，
后者仅适用于下述 gamma decoder。

- `--ottc_decoder greedy`：原行为，默认值。
- `--ottc_decoder gamma_greedy_length`：M 来自原 greedy 非 pad/unk 的 run 数，包含 silence；不使用参考。
- `--ottc_decoder gamma_reference_length`：M 来自有效训练 target 的位置数；仅作 oracle-length 诊断。

alpha 使用整句 softmax，总量恒为 1，因此没有独立长度预测。本次没有添加 quantity head 或重新训练。
不能直接用最小聚合 CE 在所有 M 中挑长度：重复位置可能使损失相同或更低。

## TIMIT 实验

复用 `reproduction/timit_ssl_native10_20261019/models/` 的 mono CE 和 upstream-JSD
已完成模型。使用历史 dev 选择的 `best_dev.pt`，同一前向同时生成三种解码结果。
全量 dev 400 句、core-test 192 句，FP32 batch1，不更新参数，沿用历史 PyTorch 后端默认值（matmul TF32 关闭，cuDNN TF32 开启）。
沿用原始 manifest/vocab/reference/normalize：去掉 sil/q，dx 映射到 d，逐句编辑距离汇总 PER。
M 包含训练时的 silence 位置，在 no-sil 打分前计算；参考长度诊断没有使用参考音素身份或时间。

```bash
cd /work/gm64/m64000/OTTC
source reproduction/libri960_continue5_20261017/environment.sh
export WANDB_MODE=disabled WANDB_DISABLED=true OTTC_EVAL_UPLOAD=0
"$LIBRI_PYTHON" reproduction/gamma_pool_timit_20261022/test_gamma_decode.py
"$LIBRI_PYTHON" reproduction/gamma_pool_timit_20261022/evaluate.py --device cuda
# 或 qsub reproduction/gamma_pool_timit_20261022/run.pbs
```

`results/` 是首次 logits 评测及 SpeechBrain 修正版；
`results_aggregation_comparison/` 是同一次前向下 logits 与 posterior 两种聚合的完整五路比较。
它们保存 RESULTS.json/CSV、逐句 predictions.jsonl、SpeechBrain 风格的 `.wer.txt`
三行对齐、PROVENANCE.json 和 COMPLETE.json。脚本逐句核对原 greedy 与历史结果并保存差异审计，
并用 frozen scorer 独立核对 PER。只有 `COMPLETE.json` 且 `full_sets=true` 才是全量完成。
评测保持本地，不上传 W&B。

对已有输出可运行 `rescore.py --output PATH` 重建 SpeechBrain 原生对齐，不重新前向。
`diagnose_repeats.py` 是独立的相邻重复合并诊断，不改变逐位置输出规则。
