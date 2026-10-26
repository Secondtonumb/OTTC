# OTTC 的两种 GW 实验（2026-10-26）

本目录是新配方，不修改历史 trainer、checkpoint 或 source snapshot。
默认 TIMIT、Wav2Vec2 Large、native 10 ms、单头、原始 AG、seed 43、batch 64。
从同一原始 SSL 初始化新训练，沿用历史 40 epoch 的 LR/warmup；首轮只跑前 5 epoch。
这不是从历史最佳模型微调。早先 AA–AE–AA 的 Libri/WavLM 诊断与此模型不同。

## 对照

| 配方 | 计划 | GW 权重 |
|---|---|---:|
| 历史 `ag__seed43` 的 epoch 5 dev 产物 | 原单调计划 | 0 |
| `monotone_w0.1__seed43` | 原单调计划，不更改 AG 主损失 | 0.1 |
| `asot_w0.1__seed43` | 可反传的 ASOT-inspired 25 步 mirror descent | 0.1 |
| `asot_w0__seed43` | 同上，用于隔离更换 solver 的效果 | 0 |

不重训历史 CE。首轮比较仅使用 development，单 seed 的 5 epoch 仍在 500-step warmup 内，不能代表收敛性能。

## 数学与原仓库的区别

来源：https://github.com/mingu6/action_seg_ot/blob/main/src/asot.py
检视的 Git blob 为 `e6645ff61ed12e964907b34c7249a15d5d5add02`。
原仓库邻域卷积及类别不等关系启发了本实现；这里独立实现 prefix sum。

令 A 是有效帧中 `0 < |t-t'| <= 2` 的邻接矩阵，B 是 occurrence 不等矩阵。
R(Q) = 0.5 * N^2 / sum(A) * <Q, A Q B>。
窗口是 native 10 ms 的左右各 20 ms；排除自身与 padding，每句独立计算后 batch sum。
目标列是出现位置：重复音素仍是独立列。只支持单头，未自动扩展到 S3T2 子状态。

方案 2：直接从原 bucketized OT 的 indices/weights 重建 Q，L = L_AG + 0.1 R(Q)。
AG 主损失与其历史数值/梯度路径保持不变。R 对 frame mass 有梯度；对 phone logits 没有直接梯度。
单调计划本身已连续，所以它能否改善自由解码碎片必须实测，不能由正则形式保证。

方案 3：row marginal 固定为模型学到的 a，column marginal 对原 phone-uniform b 使用 generalized KL。
L = <Q,C_AG> + lambda R(Q) + 1.0 KL(Q^T 1 || b)
    + 0.05 sum Q(log Q - 1) + 0.1 <Q, positional_distance>。
从 a b^T 开始，log-domain mirror descent 25 步，step_size=0.1；每步精确做 row projection。
目标、梯度和最终训练损失系数一致；对所有迭代反传，包括 logits 和 a。
这是有限步 ASOT-inspired 适配，不是声称已求得最优 GW，也不是原作者 no_grad 伪标签 + CE 训练。
AG 调用原函数，保留其 target epsilon。ASOT cost/solver 在 FP32 中运行；主干继续 BF16 autocast。
soft positional prior 不保证单调，也不保证每个 occurrence 的 argmax 覆盖。

两种 R 都作用于原始传输质量，不把 Q 偷换为预测后验。质量集中可能降低 R 而未改善预测，
所以保存 effective_frame_fraction/max_frame_mass、column_l1、backward_fraction、occurrence_coverage。
本配方没有额外 CE、CTC、forward-sum 或后验 smoothing 辅助损失。

## 验证、训练与恢复

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" -m pytest -q tests/test_gw_transport.py reproduction/timit_gw_20261026/test_integration.py
qsub -v GW_RECIPE=monotone_w0.1__seed43 reproduction/timit_gw_20261026/run.pbs
qsub -v GW_RECIPE=asot_w0.1__seed43 reproduction/timit_gw_20261026/run.pbs
qsub -v GW_RECIPE=asot_w0__seed43 reproduction/timit_gw_20261026/run.pbs
```

每条 PBS 独占一个节点/GPU，不做跨配方梯度同步。先跑真实 longest64/representative64/tail12 的
前向、反向、optimizer step，核对 SSL 初始化、所有训练组梯度和实际保存的 FP32 权重；
再做小样本原 K2 FA / text-free evaluation。profile gate 完全匹配源码/config 才运行 pilot。
若 batch64 profile OOM 或梯度失败，任务停止，不在失败配置上直接训练。

训练每个完整 epoch 原子保存独立 FP32 权重，保留所有 epoch；另外保留 best-dev 和最新完整 resume。
pilot 结束记录 PAUSED/LATEST_STATE，不能标记 40 epoch TRAIN_COMPLETE。
继续训练可使用同一 `train --recipe NAME`（不带 --pilot），自动加载最新 optimizer/scheduler/RNG/W&B ID。
40 epoch 完成后才比较 best-dev/final/final5/final10；平均累积为 FP32，成员路径/epoch/step/hash 全记录。
开发集选配置，测试集单独报告。

训练标量用 W&B；profile、所有评估、预测、音频和表格均只保存在本地。
历史 scorer 继续分别给 text-free 与 supplied-phone FA 的 P/R/F1/R-value。
无标签 forward 不求任何 reference-conditioned 计划，识别始终是 audio-only argmax。

## 碎片诊断

从原始保存的 frame IDs 计算 A–B–A 次数、middle <=20 ms 次数、短 run；这些模式本身不等于错误。
同一份 text-free hypothesis/reference 计算 S/I/D、PER，以及 gold <=30 ms 音素在确定性
最小编辑对齐中的删除率（遇到最优路径并列时依赖 tie-break）。不修改预测、不使用 oracle 删除。
报告 `PILOT_COMPARISON.json` 只比较完整 epoch5 dev 产物，保留未完成状态。
