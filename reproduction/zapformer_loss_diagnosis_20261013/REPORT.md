# Zapformer loss 排查记录（2026-10-13）

用户关注四组 W&B `train/loss`。本次只读取原训练日志和 checkpoint；
四组正式训练、原始 loss、优化器、代码快照均未修改，也未取消任务。
`inputs.json` 固定了五个 checkpoint，GPU 报告记录其完整 SHA256。

## 结论与证据范围

目前不能把真实模型的慢收敛直接归因于 SP，也不能声称单独修改辅助
正则即可解决。已确认曲线统计、共同起点的 mass 集中，以及训练辅助
loss 与状态解码之间的差异；它们需要分开处理。

### 1. 原始 train/loss 是动态批次的八卡总和

续训期间每步约 171–697 句。总 loss 会随批次句子数变化；原始总和
不是固定单位的学习曲线。按完整 epoch 计算原始 OTTC sum / 句子数，
本次日志快照得到：

| 实验 | 完整 epoch 范围 | OTTC / 句，首→末 | 下降 |
| --- | --- | --- | --- |
| JSD only | 11–21 | 6.0864 → 5.9982 | 1.45% |
| JSD + forward-sum | 11–22 | 6.0863 → 5.9266 | 2.62% |
| CE + forward-sum | 11–19 | 4.9013 → 4.4641 | 8.92% |
| CE only | 11–21 | 4.8998 → 4.3383 | 11.46% |

这些是每组内部的趋势，不是不同 objective 原始数值之间的排名。
末轮不同是运行进度不同；部分 epoch 未用于首末比较。
`loss_diagnosis.png` 同时展示原始总和、归一化分量、固定 dev400 loss
及识别 PER。`log_report.json` 保留原始 steps、epochs 和输入日志 hash。
所有派生统计标记为 backfill，不插值，不将上传时间当训练时间。

### 2. 总 loss 混合了两种 reduction

OTTC 按本卡句子求和；当前 Charsiu 按 target length 归一化后，在本卡
batch 内求平均。日志再跨八卡求和。因此 `forward_sum_raw` 是八个
本卡均值之和，不能再简单除以全局句子数来解释成每句 FS。
FS 的有效相对权重约随本卡 batch size 的倒数变化。
dev loss 每卡逐句处理，累计的 FS 又是逐句均值之和，与训练总 loss
含 FS 时不能直接比较。算法复现正确不意味着任务尺度已经校准。

### 3. 共同的 epoch10 起点有严重 mass 集中

在固定的四句真实 Libri100 样例上，原 checkpoint 单帧 mass 最大值
为 1.000、0.876、0.985、0.941；有效帧比例约 0.2%–0.3%。
这意味着这几个样例的 transport 几乎集中在一个帧上，难以为整段
音频提供充分的序列监督。不能把这四句统计当成全数据集统计。

后续 checkpoint 在同一批样例的最大 mass 已降到约 0.03–0.05，
有效帧比例约 9%–17%。所以不能声称当前四组仍全部只使用一个帧；
它们有恢复迹象，但共同起点的退化是真实存在的。

### 4. 原生辅助正则与 phone loss 的尺度不匹配风险

同四句、同模型、同 RNG、native train mode BF16、无 SpecAugment，
比较 aux_loss_scale=0 与原训练尺度。epoch10 的 encoder 辅助梯度增量
范数约为基准梯度的 16.46 倍，frontend 为 28.54 倍。
基准仍保留原生正交惩罚；该增量只度量由 aux_loss_scale 控制的
ScaleLimiter / CorrelationLimiter，不能称为去掉全部正则的纯主任务梯度。
原生辅助项按帧累计，而 OTTC 是每句质量归一化的 transport cost；
直接沿用 ASR 的 .05 不能自动保证相同任务相对尺度。

进一步从同 epoch10 权重及 Rubik 状态，对固定四句训练 120 步：
原 aux=.05：总 loss 37.9704 → 36.9951；
按平均帧数缩小 aux：37.9704 → 36.8597。
两者均仍难以拟合；缩小 aux 没有单独解决问题。因此梯度比例是
校准风险的证据，不是已证明的唯一根因。该短程测试无 SpecAugment、
单卡固定小批次，不等同于正式八卡 Libri100 实验。

### 5. Charsiu 分数适配不约束 SP 内部状态顺序

当前 score_source 合并 S0/S1/S2 概率，再对 transcript positions
归一化。将每个音素的三状态反序后，FS loss 保持 5.168386；
将 acoustic blank logit 加 20 后也保持相同，blank 梯度最大约
1.85e-10。位置 CTC 的 blank 概率由固定 -1 log-score 决定，为
0.268941，与 acoustic blank posterior 不同。

所以此 FS 监督音素位置，不直接监督 acoustic blank 或 S0→S2 顺序。
原 OTTC 仍提供三状态序列监督，不能据此说整个训练丢失了 SP 监督。
但 text-free 识别仍采用原 S3T2 greedy complete，需要完成状态路径，
该差异可能解释 deletion / insertion 恶化；还需独立实模型对照判断。
不应为美化 PER 而悄悄改变历史解码／评分协议。

### 6. SP 数值实现可以收敛，但实模型瓶颈未定

直接调用原始 archived OTTC，在自由 logits/masses 的可控单句上：
三状态 JSD：6.2031 → 0.1864；三状态 CE：5.7268 → 0.0506（200步）。
单状态 JSD：6.1756 → 0.0841；单状态 CE：4.6369 → 0.0190。
这是 Adam .15 的合成诊断，不是 Zapformer 训练结果；它只能排除
“三状态 OTTC 代码必然无法下降”，不能证明 SP 对真实数据无影响。

## 复现及跟踪

```bash
source reproduction/zapformer_s3t2ss_native_20261012/environment.sh
stage=reproduction/zapformer_loss_diagnosis_20261013
"$MFA_PYTHON" "$stage/analyze_logs.py"
"$MFA_PYTHON" "$stage/invariance.py"
qsub "$stage/probe.pbs"
qsub "$stage/overfit.pbs"
```

GPU probe：3540696.opbs，正常退出；fixed-four 对照：3540705.opbs。
诊断各自记录 W&B，正式四组保留各自原 run ID。
`publish.py` 只将派生图和证据上传到已结束的独立诊断 run，
不并发向四个正式训练 run 写历史。

此前 DDP 验证证明了计算和断点恢复可复现，未证明优化能良好收敛。
继续改进需要补充收敛测试，不应把一致性检查当作训练效果验收。
若进行无 SP、mass 温度或均匀 mass 对照，必须明确记录它们改变了
OTTC 的目标表示或输入质量；原 divergence 公式保持导入复用。
用户此前要求 OTTC 部分不改，实施这些额外对照前需要确认范围。
