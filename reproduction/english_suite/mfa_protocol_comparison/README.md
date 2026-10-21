# OTTC 与归档 CTM：统一评分与系统身份核对

更新：2026-10-09T12:34:37+09:00。

## 论文 MFA 3.0：引用值与归档身份纠正

此前将 `IF-MDD/TIMIT_MFA` 目录中的归档 CTM 标为“原生 MFA topline”是错误的：文件数值可重算，不代表模型身份已核实。下面是用户指定论文 Table II 的原文数值。

| 论文系统 | Dev TSE ms ↓ | Dev R-value % ↑ | Test TSE ms ↓ | Test R-value % ↑ |
|---|---:|---:|---:|---:|
| MFA 3.0 | 30.73 | 87.83 | 31.95 | 87.44 |
| Kaldi TDNN-F | 45.39 | 63.59 | 45.41 | 63.53 |

归档 CTM 使用 closure→stop 合并和当前冻结 scorer 后，dev/test 四项数值均在两位小数上与论文 Kaldi TDNN-F 行一致（45.39/63.59、45.41/63.53）。它的模型来源仍待核实，不能沿用目录名称认定为 MFA 3.0。

论文 MFA 的 F1 未报告，原始输出尚未找到，未完成同一评分口径的本地复现。表中论文值仅为引用；本地 OTTC 使用 TIMIT 训练并按 TIMIT dev PER 选 checkpoint，论文神经模型使用 LibriSpeech 训练，因此这些数值不作为严格公平排名。此前“OTTC 超过原生 MFA topline”的解释撤回。

[论文 Table II](https://arxiv.org/html/2609.13694v1#S4.T2) · [网页快照与归档 CTM 身份核对](../paper_mfa_audit/audit.json)。

冻结并直接复用当前 IF-MDD raw-PHN scorer 与原始 PHN reference，同时重算系统身份待核实的归档 CTM 和已完成的 K2-FSA alignment 输出。不训练模型、不重新对齐、不调整时间偏移。

归档 CTM 历史 test F1=57.22%，当前 scorer 对相同输出重算为57.36%；TSE仍精确复现47.625496ms。当前 boundary_rvalue.py 的候选匹配顺序与历史逐句结果存在差异，旧版本源码未归档，因此保留历史值，比较表统一使用本次冻结 scorer 对所有模型的重计分。

相同规则：dx→d、closure→silence、过滤 silence、保留相邻重复 phone、非静音 phone 间前一 phone 的 end 作为边界；沿用原双向匹配 PRF/R-value；hyp 不额外舍入。TSE 为每条句子的平均 |start误差|+|end误差|，再对句子取平均；仅等长序列参与。

## TIMIT core test192：统一边界指标

| 模型 / checkpoint | head | 条数 | P@20 % | R@20 % | F1@20 % | R-value % | TSE ms | TSE条数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ce/final_epoch40 | ifmdd_arpabet | 192 | 58.76 | 58.76 | 58.76 | 64.80 | 42.79 | 192 |
| ce/final_epoch40 | ottc_arpabet | 192 | 58.83 | 58.79 | 58.81 | 64.85 | 42.03 | 188 |
| ce/last3_best3_weight_average | ifmdd_arpabet | 192 | 58.86 | 58.86 | 58.86 | 64.89 | 42.81 | 192 |
| ce/last3_best3_weight_average | ottc_arpabet | 192 | 59.03 | 58.99 | 59.01 | 65.02 | 41.96 | 188 |
| ce/step_2774 | ifmdd_arpabet | 192 | 58.83 | 58.83 | 58.83 | 64.86 | 42.71 | 192 |
| ce/step_2774 | ottc_arpabet | 192 | 58.87 | 58.83 | 58.85 | 64.88 | 41.95 | 188 |
| ce/step_2847_dev_best | ifmdd_arpabet | 192 | 58.97 | 58.97 | 58.97 | 64.98 | 42.69 | 192 |
| ce/step_2847_dev_best | ottc_arpabet | 192 | 59.01 | 58.97 | 58.99 | 65.01 | 41.92 | 188 |
| ce/step_2920 | ifmdd_arpabet | 192 | 58.70 | 58.70 | 58.70 | 64.75 | 42.77 | 192 |
| ce/step_2920 | ottc_arpabet | 192 | 58.82 | 58.77 | 58.79 | 64.84 | 42.05 | 188 |
| ctc/final_epoch40 | ifmdd_arpabet | 192 | 45.70 | 45.70 | 45.70 | 53.65 | 96.01 | 192 |
| ctc/final_epoch40 | ottc_arpabet | 192 | 45.50 | 45.47 | 45.49 | 53.48 | 94.94 | 188 |
| ctc/last3_best3_weight_average | ifmdd_arpabet | 192 | 45.65 | 45.65 | 45.65 | 53.61 | 95.80 | 192 |
| ctc/last3_best3_weight_average | ottc_arpabet | 192 | 45.45 | 45.42 | 45.43 | 53.43 | 94.79 | 188 |
| ctc/step_2774 | ifmdd_arpabet | 192 | 45.58 | 45.58 | 45.58 | 53.55 | 95.57 | 192 |
| ctc/step_2774 | ottc_arpabet | 192 | 45.34 | 45.31 | 45.32 | 53.34 | 94.55 | 188 |
| ctc/step_2847 | ifmdd_arpabet | 192 | 45.70 | 45.70 | 45.70 | 53.65 | 95.98 | 192 |
| ctc/step_2847 | ottc_arpabet | 192 | 45.47 | 45.43 | 45.45 | 53.45 | 94.88 | 188 |
| ctc/step_2920_dev_best | ifmdd_arpabet | 192 | 45.70 | 45.70 | 45.70 | 53.65 | 95.99 | 192 |
| ctc/step_2920_dev_best | ottc_arpabet | 192 | 45.48 | 45.45 | 45.47 | 53.47 | 94.90 | 188 |
| jsd_upstream/final_epoch40 | ifmdd_arpabet | 192 | 63.54 | 63.54 | 63.54 | 68.88 | 38.40 | 192 |
| jsd_upstream/final_epoch40 | ottc_arpabet | 192 | 63.87 | 63.83 | 63.85 | 69.15 | 37.33 | 188 |
| jsd_upstream/last3_best3_weight_average | ifmdd_arpabet | 192 | 63.61 | 63.61 | 63.61 | 68.94 | 38.32 | 192 |
| jsd_upstream/last3_best3_weight_average | ottc_arpabet | 192 | 64.00 | 63.95 | 63.97 | 69.26 | 37.24 | 188 |
| jsd_upstream/step_2774_dev_best | ifmdd_arpabet | 192 | 64.01 | 64.01 | 64.01 | 69.28 | 38.14 | 192 |
| jsd_upstream/step_2774_dev_best | ottc_arpabet | 192 | 64.36 | 64.31 | 64.33 | 69.56 | 37.03 | 188 |
| jsd_upstream/step_2847 | ifmdd_arpabet | 192 | 63.54 | 63.54 | 63.54 | 68.88 | 38.39 | 192 |
| jsd_upstream/step_2847 | ottc_arpabet | 192 | 63.84 | 63.79 | 63.81 | 69.12 | 37.34 | 188 |
| jsd_upstream/step_2920 | ifmdd_arpabet | 192 | 63.54 | 63.54 | 63.54 | 68.88 | 38.40 | 192 |
| jsd_upstream/step_2920 | ottc_arpabet | 192 | 63.85 | 63.81 | 63.83 | 69.13 | 37.35 | 188 |
| TIMIT_MFA | 归档 ARPA（身份待核实） | 192 | 57.18 | 57.53 | 57.36 | 63.53 | 47.63 | 156 |

上表边界全部192条；TSE按每个系统可计分条数展示，不能跨不同子集直接比较。TIMIT仅保留三个 checkpoint，best3与last3为同一集合；dev_best沿用已冻结的 dev PER选择，未按此表的 test成绩重新挑选。

## 同一152条等长子集：TSE对照

既有归档 CTM 可计TSE的156条中，原始OTTC head有4条因音素映射/相邻重复融合而和raw PHN数量不等；IF-MDD head在这156条均等长。为让全部head用同一样本，进一步取交集152条；所有系统均在此152条上计TSE，边界主表仍使用完整192条。等长按位置配对，不要求音素标签序列完全相同，因此仍可能包含音素序列与标注差异。原始OTTC目标未重新映射或重训。

| 模型 / checkpoint | head | 条数 | TSE ms | ACC20 % | ACC50 % |
|---|---|---:|---:|---:|---:|
| ce/final_epoch40 | ifmdd_arpabet | 152 | 42.25 | 33.51 | 88.30 |
| ce/final_epoch40 | ottc_arpabet | 152 | 41.83 | 33.84 | 89.24 |
| ce/last3_best3_weight_average | ifmdd_arpabet | 152 | 42.27 | 33.53 | 88.39 |
| ce/last3_best3_weight_average | ottc_arpabet | 152 | 41.76 | 33.99 | 89.29 |
| ce/step_2774 | ifmdd_arpabet | 152 | 42.19 | 33.52 | 88.50 |
| ce/step_2774 | ottc_arpabet | 152 | 41.74 | 33.86 | 89.35 |
| ce/step_2847_dev_best | ifmdd_arpabet | 152 | 42.16 | 33.80 | 88.50 |
| ce/step_2847_dev_best | ottc_arpabet | 152 | 41.73 | 34.06 | 89.30 |
| ce/step_2920 | ifmdd_arpabet | 152 | 42.21 | 33.42 | 88.46 |
| ce/step_2920 | ottc_arpabet | 152 | 41.81 | 33.81 | 89.31 |
| ctc/final_epoch40 | ifmdd_arpabet | 152 | 95.81 | 7.50 | 40.12 |
| ctc/final_epoch40 | ottc_arpabet | 152 | 94.81 | 7.63 | 40.65 |
| ctc/last3_best3_weight_average | ifmdd_arpabet | 152 | 95.59 | 7.57 | 40.19 |
| ctc/last3_best3_weight_average | ottc_arpabet | 152 | 94.65 | 7.68 | 40.69 |
| ctc/step_2774 | ifmdd_arpabet | 152 | 95.42 | 7.58 | 40.58 |
| ctc/step_2774 | ottc_arpabet | 152 | 94.46 | 7.67 | 41.11 |
| ctc/step_2847 | ifmdd_arpabet | 152 | 95.80 | 7.50 | 40.28 |
| ctc/step_2847 | ottc_arpabet | 152 | 94.76 | 7.63 | 40.76 |
| ctc/step_2920_dev_best | ifmdd_arpabet | 152 | 95.81 | 7.51 | 40.12 |
| ctc/step_2920_dev_best | ottc_arpabet | 152 | 94.77 | 7.61 | 40.65 |
| jsd_upstream/final_epoch40 | ifmdd_arpabet | 152 | 38.20 | 38.08 | 92.52 |
| jsd_upstream/final_epoch40 | ottc_arpabet | 152 | 37.46 | 39.09 | 93.35 |
| jsd_upstream/last3_best3_weight_average | ifmdd_arpabet | 152 | 38.14 | 38.27 | 92.58 |
| jsd_upstream/last3_best3_weight_average | ottc_arpabet | 152 | 37.38 | 39.39 | 93.31 |
| jsd_upstream/step_2774_dev_best | ifmdd_arpabet | 152 | 37.91 | 38.90 | 92.89 |
| jsd_upstream/step_2774_dev_best | ottc_arpabet | 152 | 37.18 | 39.69 | 93.47 |
| jsd_upstream/step_2847 | ifmdd_arpabet | 152 | 38.19 | 38.04 | 92.60 |
| jsd_upstream/step_2847 | ottc_arpabet | 152 | 37.46 | 39.03 | 93.33 |
| jsd_upstream/step_2920 | ifmdd_arpabet | 152 | 38.19 | 38.12 | 92.51 |
| jsd_upstream/step_2920 | ottc_arpabet | 152 | 37.50 | 39.00 | 93.33 |
| TIMIT_MFA | 归档 ARPA（身份待核实） | 152 | 47.40 | 22.15 | 86.56 |

归档导出记录的 target 为 word_transcript，原始模型与输入构图来源未核实；OTTC/CTC K2 FA 输入为参考phone序列。这是统一评分的归档诊断表，不是论文 MFA 3.0 的复现或严格同输入比较。归档 CTM 没有 text-free PER；已有 text-free 识别/分段继续单独报告。

主表为既有 scorer 的closure→silence协议，没有把OTTC的sil预测擅自移动到stop上。所有dev/test、全部容差、label-aware边界、per-utterance与hash见JSON/CSV。L2-ARCTIC的历史MFA参考已找到，但本轮四头尚无相匹配的K2输出，因此未混入这张TIMIT表。

[完整矩阵](matrix.csv) · [逐句结果](per_utterance.csv) · [评分与输入来源](provenance.json) · [论文 MFA 与归档身份核对](../MFA_TOPLINE.md) · [text-free JSD结果](../JSD_RESULTS.md)。

W&B：[jsd_upstream](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/xlcrxd5x) · [ce](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/b23dathy) · [ctc](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/jpemos1o) · [mfa](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/0mmuz8z3)
