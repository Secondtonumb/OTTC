# 当前实验结果与测试覆盖

更新：2026-10-12T03:49:38+09:00。

[三个 checkpoint 的逐个测试与权重平均比较](../checkpoint_comparison/README.md)。

[✅ / 🟡 / ⏳ 已跑实验标记](EXPERIMENT_STATUS.md) · [声学前端差异](FRONTENDS.md)。

[论文 MFA 3.0 与归档 CTM 身份核对](MFA_TOPLINE.md) · [当前重点：OTTC 上游 JSD](JSD_RESULTS.md)。

[CE/JSD/CTC 与归档 CTM 统一重计分](mfa_protocol_comparison/README.md)：完整 test192 的 F1/R-value 与各系统共同等长子集的 TSE。

四头 TIMIT 三组已完成最终 core test 和 K2-FSA forced alignment；单头五组已完成原 README、core test、Buckeye、L2-ARCTIC。LibriSpeech 各组完成状态按实际文件标记；MFA 中途 dev/test 与最终结果分开。不同 phoneset、测试范围与 loss 定义不可混成同一排名。

## 论文 MFA 3.0：引用值与归档身份纠正

此前将 `IF-MDD/TIMIT_MFA` 目录中的归档 CTM 标为“原生 MFA topline”是错误的：文件数值可重算，不代表模型身份已核实。下面是用户指定论文 Table II 的原文数值。

| 论文系统 | Dev TSE ms ↓ | Dev R-value % ↑ | Test TSE ms ↓ | Test R-value % ↑ |
|---|---:|---:|---:|---:|
| MFA 3.0 | 30.73 | 87.83 | 31.95 | 87.44 |
| Kaldi TDNN-F | 45.39 | 63.59 | 45.41 | 63.53 |

归档 CTM 使用 closure→stop 合并和当前冻结 scorer 后，dev/test 四项数值均在两位小数上与论文 Kaldi TDNN-F 行一致（45.39/63.59、45.41/63.53）。它的模型来源仍待核实，不能沿用目录名称认定为 MFA 3.0。

论文 MFA 的 F1 未报告，原始输出尚未找到，未完成同一评分口径的本地复现。表中论文值仅为引用；本地 OTTC 使用 TIMIT 训练并按 TIMIT dev PER 选 checkpoint，论文神经模型使用 LibriSpeech 训练，因此这些数值不作为严格公平排名。此前“OTTC 超过原生 MFA topline”的解释撤回。

[论文 Table II](https://arxiv.org/html/2609.13694v1#S4.T2) · [网页快照与归档 CTM 身份核对](paper_mfa_audit/audit.json)。

## TIMIT：OTTC 与系统身份待核实的归档 CTM 重计分

复用已训练的四头模型和已有 K2-FSA 对齐输出，CE 没有重训。下表展示原始 OTTC ARPAbet head；边界 F1/R-value 使用完整 core test 192 条，TSE 使用所有系统、两个 ARPAbet head 共同等长的 152 条子集。归档 CTM 行不是论文 MFA 3.0 的复现。

| 模型 | checkpoint 选择 | FA F1@20ms %（192条）↑ | R-value %（192条）↑ | TSE ms（共同152条）↓ |
|---|---|---:|---:|---:|
| OTTC 上游 JSD | dev PER 最佳单个（step 2774） | 64.33 | 69.56 | 37.18 |
| OTTC 上游 JSD | 最后3个 = 最佳3个权重平均 | 63.97 | 69.26 | 37.38 |
| OTTC CE | dev PER 最佳单个（step 2847） | 58.99 | 65.01 | 41.73 |
| OTTC CE | 最后3个 = 最佳3个权重平均 | 59.01 | 65.02 | 41.76 |
| CTC | dev PER 最佳单个（step 2920） | 45.47 | 53.47 | 94.77 |
| CTC | 最后3个 = 最佳3个权重平均 | 45.43 | 53.43 | 94.65 |
| 归档 TIMIT_MFA CTM（系统身份待核实） | 当前 scorer 重计分；不是论文 MFA 3.0 | 57.36 | 63.53 | 47.40 |

单个 checkpoint 沿用已冻结的 dev PER 选择，不按 test 结果挑选。仅保留 step2774、2847、2920 三个 checkpoint，因此 last3 与 best3 是同一组权重平均；全部逐个测试见完整矩阵。上游 JSD 沿用原 OTTC 实现，区别于数学标准 JSD。

直接复用并冻结 IF-MDD raw-PHN scorer 和原始 PHN，源码 hash 已记录。评分采用 dx→d、closure→silence、过滤 silence，以相邻非静音 phone 间前一 phone 的 end 作为边界。TSE 是每句平均的 |start误差|+|end误差|，再对句子平均；它与后文单侧 start MAE 的定义不同，也尚未核实与论文在样本、边界提取和聚合方式上完全一致。

归档 CTM 历史 F1 为 57.22%，同一输出经当前冻结 scorer 重算为 57.36%；差异来自边界候选匹配顺序的版本变化，历史值仍保留。表中所有系统均用本次 scorer 重算。归档 CTM 自身等长子集为156条；共同子集进一步排除原 OTTC head 因映射/相邻重复融合而音素数量不等的4条。

归档导出记录的 target 为 word_transcript，模型与输入构图来源尚待核实；OTTC/CTC K2 FA 使用参考音素序列。归档 CTM 没有 text-free PER。论文 MFA 3.0、这里的归档 CTM、后文使用 MFA 标签训练的 Zapformer 分开报告。

[两个 ARPAbet head 的完整对照与 W&B](mfa_protocol_comparison/README.md) · [评分源码、输入与子集来源](mfa_protocol_comparison/provenance.json)。后文保留原评测协议的 text-free 与 K2 FA 指标，其边界定义不同，应在各自表内比较。

**归档 CTM 数值核验与身份限制：** 原始 `IF-MDD/TIMIT_MFA/ctm.phone.timit.test` 的192条输出与归档 JSONL 的标签、时间一致；冻结 scorer 和原始 PHN 的 hash 一致，重新计分复现上表。此核验只证明数值，不能证明它来自 MFA；论文身份对照见上节。

共同152条的归档 CTM 起点平均绝对误差为 24.54ms，终点为 22.86ms，相加才是表中的 47.40ms TSE。历史完整等长156条的 TSE 为 47.63ms。等长子集按位置配对，不要求 phone 标签序列完全相同；归档 CTM 在共同152条中只有91条归一化标签序列与参考完全一致，TSE仍包含音素序列差异。

[原始 CTM、hash 与数值重算证据（不验证模型身份）](mfa_protocol_comparison/MFA_ROW_VERIFIED.json)。

[W&B：历史 dev R-value 曲线与 test/checkpoint 补算表](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/dobawf16) · [保留原 epoch/step 的 CSV](rvalue_backfill/dev_rvalue.csv)。历史曲线为 P/R 派生回填，时间轴不使用回填时间。

R-value 已补齐：历史缺失值由同一条记录的 Precision/Recall 计算，直接复用冻结 IF-MDD 公式；不修改边界提取或匹配，也不重训/重推理。训练 dev、最终 test、text-free 与 K2 FA 分别报告；常规 scorer 和上方 raw-PHN scorer 的边界协议仍有区别。

## 四头 TIMIT：最终 core test 192

| 目标 | head | 有标签条数 | Text-free PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ | K2 start MAE ms ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ctc | ottc_arpabet | 192 | 10.26 | 52.92 | 19.95 | 51.92 | 58.97 | 42.18 |
| ctc | ifmdd_arpabet | 192 | 9.00 | 53.05 | 20.69 | 51.69 | 58.99 | 42.04 |
| ctc | mfa_ipa | 158 | 19.55 | — | — | — | — | — |
| ctc | zipa_ipa | 192 | 9.49 | — | — | — | — | — |
| ce | ottc_arpabet | 192 | 9.91 | 63.72 | 68.69 | 60.78 | 66.53 | 20.43 |
| ce | ifmdd_arpabet | 192 | 9.83 | 63.26 | 68.53 | 60.40 | 66.35 | 20.73 |
| ce | mfa_ipa | 158 | 11.25 | — | — | — | — | — |
| ce | zipa_ipa | 192 | 12.97 | — | — | — | — | — |
| jsd_upstream | ottc_arpabet | 192 | 9.68 | 67.62 | 72.07 | 64.78 | 69.95 | 18.24 |
| jsd_upstream | ifmdd_arpabet | 192 | 9.54 | 67.08 | 71.83 | 64.13 | 69.50 | 18.65 |
| jsd_upstream | mfa_ipa | 158 | 11.21 | — | — | — | — | — |
| jsd_upstream | zipa_ipa | 192 | 19.91 | — | — | — | — | — |

JSD 组保留上游反向 KL 混合实现，区别于数学标准 JSD。MFA/ZIPA 四头没有真实音素区间参考，仅导出 spans，不填写分段准确率。MFA 四头按词典完整覆盖子集 158 条计分。

TIMIT 四头最近真实 train loss：CTC 0.4765、CE 0.5157、上游 JSD 1.3907；三组均完成 40 轮 / 2920 updates。loss 定义不同，不按量级排名。

[完整 TIMIT 四头报告与 loss 曲线](../multihead_timit/notion_update/README.md)。

## 单头 fused phoneme：已完成的测试

| 目标 | 测试集 | 条数 | Raw PER % ↓ | F1@20ms % ↑ | R-value % ↑ |
|---|---|---:|---:|---:|---:|
| CTC | readme_kaldi_test_240 | 240 | 8.01 | 53.93 | 22.87 |
| CTC | timit_core_test_192 | 192 | 11.13 | 54.31 | 23.68 |
| CTC | buckeye_test | 1840 | 52.16 | 47.01 | -6.85 |
| CTC | l2arctic_test | 900 | 40.85 | 47.55 | -1.51 |
| OTTC CE | readme_kaldi_test_240 | 240 | 8.61 | 62.35 | 67.63 |
| OTTC CE | timit_core_test_192 | 192 | 12.07 | 62.79 | 68.02 |
| OTTC CE | buckeye_test | 1840 | 55.42 | 51.96 | 53.28 |
| OTTC CE | l2arctic_test | 900 | 41.97 | 61.38 | 63.77 |
| OTTC upstream jsd | readme_kaldi_test_240 | 240 | 8.61 | 65.29 | 70.18 |
| OTTC upstream jsd | timit_core_test_192 | 192 | 12.07 | 65.36 | 70.25 |
| OTTC upstream jsd | buckeye_test | 1840 | 55.66 | 54.40 | 55.00 |
| OTTC upstream jsd | l2arctic_test | 900 | 41.67 | 64.57 | 66.39 |
| OTTC standard JSD | readme_kaldi_test_240 | 240 | 15.53 | 61.53 | 67.43 |
| OTTC standard JSD | timit_core_test_192 | 192 | 19.46 | 61.84 | 67.71 |
| OTTC standard JSD | buckeye_test | 1840 | 56.59 | 52.52 | 55.80 |
| OTTC standard JSD | l2arctic_test | 900 | 43.11 | 62.20 | 66.13 |
| Soft-DTW CE gamma0.1 | readme_kaldi_test_240 | 240 | 74.61 | 13.02 | 34.75 |
| Soft-DTW CE gamma0.1 | timit_core_test_192 | 192 | 76.27 | 13.75 | 35.12 |
| Soft-DTW CE gamma0.1 | buckeye_test | 1840 | 91.19 | 12.23 | 34.32 |
| Soft-DTW CE gamma0.1 | l2arctic_test | 900 | 82.31 | 13.44 | 34.97 |

单头与四头训练设置和评测范围不同。单头分段为已有 text-free 协议，未冒充 K2 FA 结果。Soft-DTW 此配置出现严重删除、结果较差，不能用 loss 下降断言收敛良好。

## LibriSpeech 当前 dev 与最终 test 状态

[W&B dev PER 与分段曲线](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/reports/English-phonemes:-dev-PER-and-segmentation--VmlldzoxODA3OTU2MQ==)。

| 规模 / 目标 | dev epoch | head | PER / token error % | Precision@20ms % | Recall@20ms % | F1@20ms % | R-value % | 最终 test |
|---|---:|---|---:|---:|---:|---:|---:|---|
| libri100/ctc | 40.0 | ottc_arpabet | 3.75 | 44.79 | 73.06 | 55.54 | 33.86 | ✅ 已完成 |
| libri100/ctc | 40.0 | ifmdd_arpabet | 3.79 | 45.00 | 73.23 | 55.74 | 34.24 | ✅ 已完成 |
| libri100/ctc | 40.0 | mfa_ipa | 2.49 | — | — | — | — | ✅ 已完成 |
| libri100/ctc | 40.0 | zipa_ipa | 2.45 | — | — | — | — | ✅ 已完成 |
| libri100/ce | 40.0 | ottc_arpabet | 4.42 | 72.77 | 71.49 | 72.13 | 76.26 | ✅ 已完成 |
| libri100/ce | 40.0 | ifmdd_arpabet | 4.99 | 72.77 | 71.11 | 71.93 | 76.10 | ✅ 已完成 |
| libri100/ce | 40.0 | mfa_ipa | 4.05 | — | — | — | — | ✅ 已完成 |
| libri100/ce | 40.0 | zipa_ipa | 3.60 | — | — | — | — | ✅ 已完成 |
| libri100/jsd_upstream | 40.0 | ottc_arpabet | 4.64 | 74.78 | 73.63 | 74.20 | 78.01 | ✅ 已完成 |
| libri100/jsd_upstream | 40.0 | ifmdd_arpabet | 5.19 | 74.81 | 73.25 | 74.02 | 77.86 | ✅ 已完成 |
| libri100/jsd_upstream | 40.0 | mfa_ipa | 4.36 | — | — | — | — | ✅ 已完成 |
| libri100/jsd_upstream | 40.0 | zipa_ipa | 3.64 | — | — | — | — | ✅ 已完成 |
| libri960/ctc | 20.21 | ottc_arpabet | 2.86 | 36.69 | 67.88 | 47.63 | 13.14 | 🟡 训练结束后自动 test |
| libri960/ctc | 20.21 | ifmdd_arpabet | 2.91 | 36.79 | 68.02 | 47.75 | 13.31 | 🟡 训练结束后自动 test |
| libri960/ctc | 20.21 | mfa_ipa | 1.44 | — | — | — | — | 🟡 训练结束后自动 test |
| libri960/ctc | 20.21 | zipa_ipa | 1.78 | — | — | — | — | 🟡 训练结束后自动 test |
| libri960/ce | 18.94 | ottc_arpabet | 5.54 | 68.35 | 68.09 | 68.22 | 72.90 | 🟡 训练结束后自动 test |
| libri960/ce | 18.94 | ifmdd_arpabet | 6.11 | 68.28 | 67.66 | 67.97 | 72.71 | 🟡 训练结束后自动 test |
| libri960/ce | 18.94 | mfa_ipa | 4.51 | — | — | — | — | 🟡 训练结束后自动 test |
| libri960/ce | 18.94 | zipa_ipa | 3.94 | — | — | — | — | 🟡 训练结束后自动 test |
| libri960/jsd_upstream | 18.79 | ottc_arpabet | 5.67 | 72.92 | 72.63 | 72.77 | 76.77 | 🟡 训练结束后自动 test |
| libri960/jsd_upstream | 18.79 | ifmdd_arpabet | 6.18 | 72.94 | 72.23 | 72.58 | 76.63 | 🟡 训练结束后自动 test |
| libri960/jsd_upstream | 18.79 | mfa_ipa | 5.40 | — | — | — | — | 🟡 训练结束后自动 test |
| libri960/jsd_upstream | 18.79 | zipa_ipa | 4.64 | — | — | — | — | 🟡 训练结束后自动 test |

## LibriSpeech 已完成的最终 text-free test

| 数据 / 目标 | head | Test PER / token error % | F1@20ms % | R-value % |
|---|---|---:|---:|---:|
| libri100/ctc | ottc_arpabet | 3.65 | 55.25 | 33.80 |
| libri100/ctc | ifmdd_arpabet | 3.67 | 55.46 | 34.15 |
| libri100/ctc | mfa_ipa | 2.41 | — | — |
| libri100/ctc | zipa_ipa | 2.36 | — | — |
| libri100/ce | ottc_arpabet | 4.41 | 71.38 | 75.63 |
| libri100/ce | ifmdd_arpabet | 4.93 | 71.16 | 75.46 |
| libri100/ce | mfa_ipa | 4.05 | — | — |
| libri100/ce | zipa_ipa | 3.52 | — | — |
| libri100/jsd_upstream | ottc_arpabet | 4.55 | 73.31 | 77.26 |
| libri100/jsd_upstream | ifmdd_arpabet | 5.06 | 73.14 | 77.12 |
| libri100/jsd_upstream | mfa_ipa | 4.34 | — | — |
| libri100/jsd_upstream | zipa_ipa | 3.56 | — | — |

LibriSpeech 时间参考来自自动对齐；中途 dev 在不同 epoch，不能作为最终公平比较。本地原名称为 eval_PER_percent/<head>、eval_boundary_F1_20ms/<head>；W&B 显示 eval/PER_percent/<head>、eval/boundary_F1_20ms/<head>。新进程同时写明确的 dev/* 别名。每组最终训练自带 test，另由 evaluate.pbs 补 K2 FA。

| Libri100 目标 | 最近 train loss | 记录 epoch | 最新训练步数 |
|---|---:|---:|---:|
| ctc | 0.06 | 39.91 | 17840 |
| ce | 0.33 | 39.91 | 17840 |
| jsd_upstream | 0.98 | 39.91 | 17840 |

## MFA English 外部 TIMIT dev/test

| 目标 | checkpoint step | 阶段 | Split | PER % | Text-free F1@20ms % | Text-free R-value % | K2 FA F1@20ms % | K2 FA R-value % |
|---|---:|---|---|---:|---:|---:|---:|---:|
| ctc | 70014 | 正式逐轮 dev | development | 41.32 | 42.30 | 13.24 | — | — |
| ctc | 70014 | 最终 | development | 41.32 | 42.30 | 13.24 | — | — |
| ctc | 70014 | 最终 | test | 41.44 | 42.74 | 13.65 | 53.73 | 60.52 |
| ottc | 70014 | 正式逐轮 dev | development | 56.41 | 49.57 | 55.07 | — | — |
| ottc | 70014 | 最终 | development | 56.41 | 49.57 | 55.07 | — | — |
| ottc | 70014 | 最终 | test | 54.96 | 51.17 | 56.77 | 42.16 | 51.59 |
| spottc | 68868 | 正式逐轮 dev | development | 44.47 | 46.65 | 7.94 | — | — |
| spottc | 68868 | 最终 | development | 44.47 | 46.65 | 7.94 | — | — |
| spottc | 68868 | 最终 | test | 45.30 | 47.58 | 8.70 | 45.65 | 53.83 |
| film_k4 | 109409 | 正式逐轮 dev | development | 48.98 | 50.74 | 10.20 | — | — |
| film_k4 | 109409 | 最终 | development | 48.98 | 50.74 | 10.20 | — | — |
| film_k4 | 109409 | 最终 | test | 48.82 | 51.38 | 10.88 | 49.27 | 56.71 |

| MFA 当前 test 目标 | K2 start MAE ms | K2 end MAE ms | K2 token IoU % | W&B |
|---|---:|---:|---:|---|
| ctc | 24.59 | 25.80 | 50.87 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/ka86ncnv) |
| ottc | 43.49 | 45.38 | 39.58 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/bs5kmlwr) |
| spottc | 58.98 | 65.09 | 25.71 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/amxu3lul) |

MFA 原先没有 dev/test logging，已经增加下一进程恢复时 dev 和每轮 dev。development 使用原配置的 TIMIT 4096 条，core test 192 条；它们没有进入 MFA 全量训练。按冻结的 TIMIT→MFA 标签映射计 PER，闭塞/释放相邻融合；该结果包含标注体系差异，与四头词典 MFA IPA 目标不同。SPOTTC/FiLM 采用原 S3T2 greedy complete；K2 FA 在三状态边缘化的 phone posterior 上执行，与原 substate topology FA 区分。

当前固定中途 checkpoint 的 MFA test PER 约 CTC 45.12%、OTTC 61.14%、SPOTTC 45.16%，效果仍差。SPOTTC 的 K2 起止 MAE 约 305/303ms、IoU 接近 0，需要单独调查时序和 substate 解释；CTC 为 32/34ms、OTTC 为 53/56ms。未对预测拟合或修正时间偏移，未基于 test 改超参或挑选 checkpoint。

## 自动保存、切换与补测

MFA 当前片段在预算到时保存 optimizer/sampler/RNG，正常退出后续训提交优化配置；新片段同时启用逐轮外部 dev。Libri100 保持当前设置完成，保存及 test 后进入 microbatch8 的 Libri960。所有新 run 沿用 W&B；test 不参与 checkpoint 选择。

MFA 数据准备 job 3512374、四模型输出一致性 gate 3512417，以及三组完整 dev4096 / core test192 / K2 test192 jobs 3512419 / 3512434 / 3512435 均已成功完成。当前诊断 test 和最终 test 分开存放，不覆盖。FiLM 正式训练尚未启动；现有 30-step 验证权重仅做 2 条推理校验，不当作正式指标。MFA CTC 已正常保存 step26266，并自动提交优化续训 job3512447；其余组在片段结束时自动切换。新的八 rank 逐轮 dev hook 等待优化续训启动后验证。最终 epoch10 自动 test。

[最新进度与 W&B](TRACKING.md) · [吞吐实测](THROUGHPUT.md) · [实验配置](README.md)。
