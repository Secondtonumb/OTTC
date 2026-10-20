四组原生 Zapformer / Libri100 实验均已完成50轮（step5300），PBS Exit_status=0。训练正常结束不等于模型已成功收敛。

不建议直接按原配置延长训练。TIMIT dev400映射PER存在跨标注体系差异，不能作为LibriSpeech WER；四组在共同数据、共同初始化、共同协议下表现出严重插入或漏删，足以支持先做受控诊断。

| arm | train OTTC/句 | dev mapped PER% | insertions% | deletions% | FA F1@20ms% | FA R@20ms% |
| --- | --- | --- | --- | --- | --- | --- |
| jsd_fs | 5.671562 | 153.17 | 120.07 | 0.97 | 41.12 | 50.06 |
| jsd_only | 5.684683 | 94.63 | 0.00 | 93.41 | 41.44 | 50.26 |
| ce_fs | 3.432762 | 109.70 | 69.02 | 2.39 | 51.38 | 58.75 |
| ce_only | 3.435014 | 91.81 | 0.00 | 89.79 | 41.70 | 50.68 |

完整dev/test及text-free与supplied-phone FA的P/R/F1/R-value、scorer/source hash、边界协议均保存在completed50_assessment.json；保持各自协议及原始epoch/step，未插值、未重训。强制对齐提供音素，text-free指标为仅音频。原始JSD与CE的loss定义不同，不能跨定义比较绝对值。

Monophone候选（尚未提交，等待用户选择）：

1. 保留313维head，logsumexp合并每个音素的三状态后得到105维音素分数，用未展开的音素序列调用同一个archived原始OTTC-JSD；mass head及归一化不变，FS=0。复用共同第10轮checkpoint/优化器可隔离目标表示与解码变化，严格标注为pooled-state monophone诊断，不冒称真正105维输出模型。
2. 真正使用105维输出head、相同Zapformer、原始OTTC-JSD、mass head，FS=0，seed42随机初始化。这是独立实验；313维checkpoint的head和优化器不能完整直接恢复。

先用单句/固定小批次验证loss下降、目标音素覆盖、识别编辑距离、mass集中程度及断点复现，再考虑Libri100正式训练。保留313状态的同checkpoint诊断若失败，不能排除checkpoint已有坏状态的影响；需要相同seed随机初始化的配对诊断。Monophone与SP具有不同类别/目标表示，不能只比较raw loss绝对值。

已训练的四组checkpoint、冻结源码与W&B历史保持原样。
