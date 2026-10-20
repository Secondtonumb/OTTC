LR 与 forward-sum 的可复现诊断

原始 OTTC divergence、SP 三状态目标、mass head、四组正式训练均未修改。诊断代码与结果仅写入本目录。

实际 LR 与更新

原生模型只有一个 LR 参数组（958 个参数张量），包含 encoder/frontend/state head/mass head，没有意外的低 LR head。原第10轮 checkpoint 的 LR 为 0.0005；用户选择重规划50轮后，第11轮 LR 为 0.017118958。正式训练的实际 LR 见 formal_snapshot.json，不能继续用旧的0.0005解释当前曲线。

固定四句、相同第10轮模型/优化器、相同 seed42、120次更新：

| LR | FS weight | 初始 JSD/句 | 最终 JSD/句 | JSD降幅 | FS local mean |
| --- | --- | --- | --- | --- | --- |
| 0.01712 | 0 | 6.082058 | 6.030655 | 0.85% | 0.0000 → 0.0000 |
| 0.01712 | 1 | 6.082058 | 6.046687 | 0.58% | 13.6422 → 12.8084 |
| 0.03424 | 0 | 6.082058 | 5.991096 | 1.50% | 0.0000 → 0.0000 |
| 0.03424 | 1 | 6.082058 | 5.905179 | 2.91% | 13.6422 → 9.9007 |

将 LR 翻倍，FS=0 时最终 JSD/句从 6.030655 改善到 5.991096；FS=1 时从 6.046687 改善到 5.905179。本次短测支持提高LR有帮助，且翻倍LR+FS组合下降最多，不能认定FS直接把JSD卡住。降幅仍有限，不能宣称JSD已完全收敛，也不能证明0.03424是正式训练的最优LR。

四组初始20步均值/最终20步均值见 report.json；结果方向与最终单点相同。原LR无FS组的第一步 encoder/frontend/state head/mass head 参数相对更新分别为 0.001245 / 0.000930 / 0.002664 / 0.001208，证实优化器并未停止更新。这些数值属于隔离诊断，不是正式DDP训练的参数差值。

所有组合的初始 OTTC loss 逐位一致。完整 first/last20均值和每步曲线见 report.json，不能只用最后一个点下结论。模型使用原生 BatchedRubik 和恢复的优化器状态，LR 在诊断中固定；未使用 SpecAugment。BF16未乘AMP loss scale，因此 auxiliary scale 为未缩放的0.05。原生 batch-count 固定在step1060，正交约束保持原生实现。本实验属于短程过拟合诊断，不等于正式8卡训练的50轮LR扫描。

四句诊断中，OTTC是local sum、FS是local mean；正式批次远大于四句，因此 FS 在诊断里的相对比例更大。这里测得的最优权重不能直接搬到正式训练。对照保留同样的原生 auxiliary 正则。

梯度关系（同一个 forward，排除 backbone 内部 auxiliary backward）：

| checkpoint | logits cosine | logits FS/OTTC norm | encoder output cosine | encoder output FS/OTTC norm |
| --- | --- | --- | --- | --- |
| origin_epoch10 | -0.001013 | 0.1961 | -0.007750 | 0.1694 |
| jsd_fs_step2250 | 0.044927 | 0.3796 | 0.005952 | 0.0669 |

两处梯度都接近正交，没有发现显著相反方向或 FS 梯度压倒 OTTC 的证据。这个结论只覆盖这两份 checkpoint 和四句固定真实样本；不代表全训练集梯度永远一致。

正式训练 matched epoch 23 的 OTTC/句：

- JSD + FS: 5.910122
- JSD only: 5.962247
- CE + FS: 4.149030
- CE only: 4.191018

JSD only 与 JSD + FS 可互比；CE only 与 CE + FS 可互比；CE 与 JSD 的 raw loss 定义不同。

固定 TIMIT dev400、相同 optimizer step 2489 的诊断（不是 LibriSpeech WER）：

| arm | mapped PER% | insertion% | deletion% | supplied-phone FA F1@20ms% | supplied-phone FA R@20ms% | text-free F1@20ms% | text-free R@20ms% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| jsd_fs | 171.78 | 125.30 | 1.10 | 41.54 | 50.37 | 36.46 | -68.11 |
| jsd_only | 92.88 | 0.01 | 89.39 | 39.11 | 48.24 | 21.38 | -524.77 |
| ce_fs | 90.36 | 32.39 | 7.76 | 57.40 | 63.68 | 30.68 | -233.45 |
| ce_only | 92.10 | 0.01 | 89.85 | 43.93 | 52.36 | 21.21 | -533.05 |

完整 P/R/F1/R-value、原始记录区间、协议与 scorer/source hash 保存在 formal_snapshot.json 和 formal_snapshot/。强制对齐提供目标音素；text-free recognition 使用音频及原生 greedy complete。边界协议为保存的 native phone-level变化/10ms grid、blank previous，以及原有冻结 R-value 公式；不改 scorer、不与 MFA 论文 topline做结论比较。本记录是 backfill，保留原始 epoch/step，不插值、不把导入时间当训练时间。

forward-sum 的具体限制

现有适配先合并 S0/S1/S2，再按目标位置取分数并重新归一化。因此它不能约束原模型 acoustic blank 的绝对概率，也不能区分三状态的顺序；CTC的blank使用独立固定分数-1。此前的 forward_sum_invariance.json已验证这两项不变性。这是经确认的现有输出适配的性质，不是修改原生OTTC的理由。

正式训练中，FS两组的OTTC下降均略快；JSD+FS同时有大量识别插入，CE+FS的FA和识别则好于CE-only。因此不能把四组loss不易下降统一归因于FS，但JSD+FS确有需要检查的识别副作用。梯度近正交也不能排除目标与解码约束不一致。

复现命令：从干净的新输出目录复制 diagnose.py 与对应PBS输出路径，再提交 `qsub diagnose.pbs`。脚本显式拒绝向已有结果追加一个从step0重跑的新实验。原始 checkpoint SHA256 为 302ff5e361fa2add46e336e63873d5597558855360cc25b8f10d950af3e2f74c。每组独立W&B run，run ID与配置记录在 cases/*/wandb_run.json；已有结果不会被重跑覆盖。
