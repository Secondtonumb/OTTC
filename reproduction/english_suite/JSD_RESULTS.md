# OTTC 上游 JSD：重点结果

更新：2026-10-12T03:49:38+09:00。

重点查看 OTTC 上游 jsd 的识别、text-free segmentation 和 IF-MDD K2-FSA forced alignment。CE 作为已有对照，同数据与模型配置已训练完成的 CE 不重训。这里保留用户指定的上游 jsd 实现，区别于数学标准 Jensen–Shannon divergence。

| 数据 / 四头 JSD | 状态 | 最近 train loss | Dev PER % | Dev F1@20ms % | Dev R-value % | W&B |
|---|---|---:|---:|---:|---:|---|
| timit | ✅ 完成40轮 | 1.39 | 8.34 | 68.79 | 73.14 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/q0zlpgjh) |
| libri100 | ✅ 完成40轮 | 0.98 | 4.64 | 74.20 | 78.01 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/459q6sop) |
| libri960 | 🟡 训练中 18.79/40 | 1.38 | 5.67 | 72.77 | 76.77 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/11ka2qzu) |

dev 表使用原始 OTTC ARPAbet head；不同阶段的 dev 不当作最终比较。

## LibriSpeech 最终 text-free test：JSD 与已有对照

| 数据 | 目标 | Test PER % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA |
|---|---|---:|---:|---:|---|
| libri100 | JSD | 4.55 | 73.31 | 77.26 | 已完成 |
| libri100 | CE（复用） | 4.41 | 71.38 | 75.63 | 已完成 |
| libri100 | CTC | 3.65 | 55.25 | 33.80 | 已完成 |

表内为原始 OTTC ARPAbet head；LibriSpeech 分段参考来自自动对齐。最终 epoch40 模型与 dev 选择/权重平均结果分开报告。

## 三个 checkpoint：JSD 与已有 CE 对照

| 数据 | 目标 / 策略 | step | Test PER % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |
|---|---|---|---:|---:|---:|---:|---:|
| timit | JSD / dev最好单个 | 2774 | 9.69 | 67.72 | 72.15 | 65.09 | 70.21 |
| timit | JSD / 最后3个权重平均 | 2774,2847,2920 | 9.72 | 67.64 | 72.08 | 64.81 | 69.97 |
| timit | JSD / dev最好3个权重平均 | 2774,2920,2847 | 9.72 | 67.64 | 72.08 | 64.81 | 69.97 |
| timit | CE（复用） / dev最好单个 | 2847 | 9.91 | 63.82 | 68.78 | 60.91 | 66.64 |
| timit | CE（复用） / 最后3个权重平均 | 2774,2847,2920 | 9.91 | 63.75 | 68.71 | 60.95 | 66.68 |
| timit | CE（复用） / dev最好3个权重平均 | 2847,2920,2774 | 9.91 | 63.75 | 68.71 | 60.95 | 66.68 |
| libri100 | JSD / dev最好单个 | 16056 | 4.63 | 73.68 | 77.57 | 62.19 | 67.95 |
| libri100 | JSD / 最后3个权重平均 | 15164,15610,16056 | 4.70 | 73.77 | 77.65 | 62.22 | 67.97 |
| libri100 | JSD / dev最好3个权重平均 | 16056,15610,15164 | 4.70 | 73.77 | 77.65 | 62.22 | 67.97 |
| libri100 | CE（复用） / dev最好单个 | 15610 | 4.44 | 71.46 | 75.70 | 58.06 | 64.49 |
| libri100 | CE（复用） / 最后3个权重平均 | 15164,15610,16056 | 4.44 | 71.42 | 75.66 | 57.92 | 64.37 |
| libri100 | CE（复用） / dev最好3个权重平均 | 15610,16056,15164 | 4.44 | 71.42 | 75.66 | 57.92 | 64.37 |

[timit JSD checkpoint 比较 W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/n2xcz9e9) · [timit CE（复用） checkpoint 比较 W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/kbka0y87) · [libri100 JSD checkpoint 比较 W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/f88q3c90) · [libri100 CE（复用） checkpoint 比较 W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/vkl0wfbj)


TIMIT 仅保留最后三个 checkpoint，因此最好三个与最后三个组成相同集合，权重平均结果相同。选最好单个和最好三个仅使用 dev PER；上表每个模型在完整 test 上分别评分。

## 原始单头 JSD（已完成）

| 目标 | README test240 PER % | Text-free F1@20ms % | Text-free R-value % |
|---|---:|---:|---:|
| 上游 JSD | 8.61 | 65.29 | 70.18 |
| CE（复用） | 8.61 | 62.35 | 67.63 |

单头与四头数据范围及训练配置不同，分别报告。MFA 标签 Zapformer 是另一实验系列，不列作本次四头 JSD 的结果。

[Zapformer JSD 支持与验证](mfa_jsd/README.md)：复用原 OTTC JSD 归档实现，完整模型 GPU 前向/反向与 loss/梯度一致性检查已通过；新 JSD 长训尚未提交。

[论文 MFA 3.0 与归档 CTM 身份核对](MFA_TOPLINE.md)：论文 test TSE=31.95ms、R-value=87.44%；原生 MFA topline 的旧标注已撤回，归档 CTM 身份仍待核实。

[CE/JSD/CTC 与归档 CTM 的统一重计分](mfa_protocol_comparison/README.md)：冻结 scorer、raw PHN、同一 test192 和各系统共同等长 TSE 子集；复用既有 K2 输出。这不是论文 MFA 3.0 的同口径复现。

[全部四头与 checkpoint 结果](../checkpoint_comparison/README.md) · [实时 dev 曲线](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/reports/English-phonemes:-dev-PER-and-segmentation--VmlldzoxODA3OTU2MQ==) · [实验状态](EXPERIMENT_STATUS.md)。
