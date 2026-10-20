# 论文 MFA 3.0 与归档 CTM 身份核对

更新：2026-10-09T12:34:36+09:00。

## 论文 MFA 3.0：引用值与归档身份纠正

此前将 `IF-MDD/TIMIT_MFA` 目录中的归档 CTM 标为“原生 MFA topline”是错误的：文件数值可重算，不代表模型身份已核实。下面是用户指定论文 Table II 的原文数值。

| 论文系统 | Dev TSE ms ↓ | Dev R-value % ↑ | Test TSE ms ↓ | Test R-value % ↑ |
|---|---:|---:|---:|---:|
| MFA 3.0 | 30.73 | 87.83 | 31.95 | 87.44 |
| Kaldi TDNN-F | 45.39 | 63.59 | 45.41 | 63.53 |

归档 CTM 使用 closure→stop 合并和当前冻结 scorer 后，dev/test 四项数值均在两位小数上与论文 Kaldi TDNN-F 行一致（45.39/63.59、45.41/63.53）。它的模型来源仍待核实，不能沿用目录名称认定为 MFA 3.0。

论文 MFA 的 F1 未报告，原始输出尚未找到，未完成同一评分口径的本地复现。表中论文值仅为引用；本地 OTTC 使用 TIMIT 训练并按 TIMIT dev PER 选 checkpoint，论文神经模型使用 LibriSpeech 训练，因此这些数值不作为严格公平排名。此前“OTTC 超过原生 MFA topline”的解释撤回。

[论文 Table II](https://arxiv.org/html/2609.13694v1#S4.T2) · [网页快照与归档 CTM 身份核对](paper_mfa_audit/audit.json)。

## 系统身份待核实的既有 TIMIT CTM 数值

以下路径与 JSON 键名保留历史名称，仅用于追溯，不能凭其中的 mfa/native_mfa 字样认定为论文 MFA 3.0。

| 协议 | Split | 边界计分条数 | TSE计分条数 | F1@20ms % | R-value % | TSE ms |
|---|---|---:|---:|---:|---:|---:|
| timit_mfa_ctm_raw_phn_tse_rvalue | timit_dev | 400 | 364 | 57.28 | 63.47 | 47.69 |
| timit_mfa_ctm_raw_phn_tse_rvalue | timit_test | 192 | 156 | 57.22 | 63.38 | 47.63 |
| timit_mfa_ctm_raw_phn_tse_rvalue_closuremerge | timit_dev | 400 | 364 | 57.28 | 63.47 | 45.39 |
| timit_mfa_ctm_raw_phn_tse_rvalue_closuremerge | timit_test | 192 | 156 | 57.22 | 63.38 | 45.41 |

TIMIT core test192 有36条归档 CTM/GT phone 数量不等，历史 TSE 只在156条等长样本计算；历史 boundary F1/R-value仍使用192条。TSE不等于当前报告的单侧 start/end MAE。

## 沿用当前 OTTC 边界评分协议重新计分

直接复用当前 scorer，读取同一 test192 manifest/reference：所有 phone interval 起止边界去重、排除音频两端20ms、按容差做一对一匹配。重用已保存的 CTM，未重跑 aligner，未修正时间偏移。

| 系统 | 条数 | F1@10ms % | F1@20ms % | F1@30ms % |
|---|---:|---:|---:|---:|
| 归档 CTM（身份待核实） | 192 | 28.18 | 53.89 | 71.65 |

此行与当前 OTTC 使用相同 utterance/reference/边界算法，但归档 CTM 的模型和输入构图来源未核实。不能凭这一行宣称超过论文 MFA，也不能从 transcript-conditioned CTM 推算 text-free PER。

当前参考含 silence/closure 区间的起止边界（共7123个），历史 no-sil scorer 的参考边界为5562个，因此当前 F1=53.89% 与历史57.22%属于不同评分口径，不能解释为模型性能变化。

L2-ARCTIC 既有 MFA10ms baseline：test900 的 F1@20ms=79.57%、R-value=82.57%；严格 TSE=29.04ms仅计778条等长样本（另122条不等长）。来源：[原始 L2-ARCTIC MFA报告](/home/m64000/work/IF-MDD/fa_research/results/l2arctic_mfa_gt_fa_10ms/summary.md)。

[CE/JSD/CTC 与归档 CTM 沿用同一个冻结 scorer 的对照](mfa_protocol_comparison/README.md)：完整 test192 边界与各系统共同等长 TSE 子集。

[TIMIT原始汇总](/home/m64000/work/IF-MDD/fa_research/results/paper_prep_experiment_audit/final_paper_tables_zh.md) · [核对JSON](mfa_topline_audit.json) · [当前JSD结果](JSD_RESULTS.md)。
