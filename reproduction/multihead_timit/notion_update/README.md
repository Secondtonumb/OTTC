# OTTC 英语音素实验：单头复现与四 phoneset 联合训练

快照时间：2026-10-08T14:53:08+09:00。本文为本地实验记录，尚未发布到 Notion。

当前范围：先用 TIMIT train，之后扩到 LibriSpeech 100h。当前三组模型分别为 CTC、OTTC+CE、OTTC+原始上游 jsd；每组共享一个编码器，同时训练四个 phoneset 头。

## 四头配置及监督来源

| Head | 输出数（含特殊标签） | Train / dev / test 有标签条数 |
| --- | --- | --- |
| 原始 OTTC ARPAbet | 42 | 4620 / 400 / 192 |
| IF-MDD ARPAbet | 43 | 4620 / 400 / 192 |
| MFA IPA | 81 | 3909 / 339 / 158 |
| ZIPA IPA127 | 127 | 4620 / 400 / 192 |

原始 OTTC 头沿用 fused mapping，保留 q、dx；IF-MDD 头沿用旧输出表，TIMIT 目标做 60→39、删除 q、dx→d，完整输出表保留未监督的 ao/zh。MFA 头采用官方 US IPA v3.1 词典的最高概率发音，不是 MFA 声学对齐；词典覆盖不足的语音仅屏蔽 MFA loss，训练覆盖率 84.61%。ZIPA 使用原生 IPA127 输出表及显式 ARPAbet→IPA 标签映射，未使用 ZIPA 模型伪标签。

底座为官方 facebook/wav2vec2-large-lv60 原始权重。TIMIT train 4620（包含 SA1/SA2）、dev 400、core test 192，三个 split 说话人不重叠。seed 42；40 epochs；有效 batch 64、microbatch 16、梯度累积 4；CNN 冻结；BF16；AdamW weight decay 0.005、warmup 500、linear decay。CTC lr=5e-5，OTTC lr=2e-4；四头等权；OTTC 完成第 30 轮后冻结对齐分支。

## 当前训练进度及 W&B

| Objective | 状态 | 更新步数 | 最新 dev epoch | 最近 train loss | W&B |
| --- | --- | --- | --- | --- | --- |
| CTC | 完成 | 2920/2920 | 40.0 | 0.4765 | https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/7tp3slt5 |
| OTTC + CE | 完成 | 2920/2920 | 40.0 | 0.5157 | https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/ql9zhyav |
| OTTC + 上游 jsd | 完成 | 2920/2920 | 40.0 | 1.3907 | https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/q0zlpgjh |

三组原始 loss 的定义/reduction 不同，不能按数值高低排名；三组均完成 40 轮、2920 个更新步，最终 core test 已完成。四头扩展实验也不是未修改的单头 README 复现。

## Loss 下降情况

![CTC、OTTC CE 与上游 jsd 的已记录训练 loss](figures/train_losses.png)

三幅图分别使用各自纵轴，仅绘制真实记录点；Markdown 是本地快照，W&B 链接用于查看在线曲线。

| Objective | 首个记录 train loss | 末个记录 train loss | 下降比例 | 累计 Trainer runtime（分钟） |
| --- | --- | --- | --- | --- |
| CTC | 14.8542 | 0.4765 | 96.79% | 31.09 |
| OTTC + CE | 4.1768 | 0.5157 | 87.65% | 37.07 |
| OTTC + 上游 jsd | 6.1399 | 1.3907 | 77.35% | 36.45 |

首个记录为 step 50 / epoch 0.68，末个记录为 step 2900 / epoch 39.73；不补造 step 0 或 2920 的训练 loss。runtime 为各训练调用实测时间之和，包含每轮评估及保存开销，不包含排队；断点续训段已相加，未使用续训时被总步数稀释的 train_loss。这些下降比例描述各自曲线，不等同于不同目标之间的性能排名或已充分收敛的证明。

| Objective | Head | 首个 train head loss | 末个 train head loss |
| --- | --- | --- | --- |
| CTC | 原始 OTTC ARPAbet | 11.7839 | 0.4190 |
| CTC | IF-MDD ARPAbet | 12.0381 | 0.3790 |
| CTC | MFA IPA | 16.7078 | 0.6957 |
| CTC | ZIPA IPA127 | 18.8869 | 0.4124 |
| OTTC + CE | 原始 OTTC ARPAbet | 3.7512 | 0.5585 |
| OTTC + CE | IF-MDD ARPAbet | 3.7367 | 0.5381 |
| OTTC + CE | MFA IPA | 4.4004 | 0.4454 |
| OTTC + CE | ZIPA IPA127 | 4.8191 | 0.5207 |
| OTTC + 上游 jsd | 原始 OTTC ARPAbet | 6.1127 | 1.5078 |
| OTTC + 上游 jsd | IF-MDD ARPAbet | 6.1014 | 1.4506 |
| OTTC + 上游 jsd | MFA IPA | 6.1668 | 1.1766 |
| OTTC + 上游 jsd | ZIPA IPA127 | 6.1788 | 1.4278 |

## Text-free inference：最终 TIMIT core test 192

| Objective | Head | 有标签语音 | 原生识别错误率 % ↓ | 边界 Precision % | 边界 Recall % | F1@20ms % ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| CTC | 原始 OTTC ARPAbet | 192 | 10.26 | 41.09 | 74.29 | 52.92 |
| CTC | IF-MDD ARPAbet | 192 | 9.00 | 41.27 | 74.23 | 53.05 |
| CTC | MFA IPA | 158 | 19.55 | — | — | 无真实参考边界 |
| CTC | ZIPA IPA127 | 192 | 9.49 | — | — | 无真实参考边界 |
| OTTC + CE | 原始 OTTC ARPAbet | 192 | 9.91 | 62.70 | 64.78 | 63.72 |
| OTTC + CE | IF-MDD ARPAbet | 192 | 9.83 | 62.89 | 63.65 | 63.26 |
| OTTC + CE | MFA IPA | 158 | 11.25 | — | — | 无真实参考边界 |
| OTTC + CE | ZIPA IPA127 | 192 | 12.97 | — | — | 无真实参考边界 |
| OTTC + 上游 jsd | 原始 OTTC ARPAbet | 192 | 9.68 | 66.57 | 68.69 | 67.62 |
| OTTC + 上游 jsd | IF-MDD ARPAbet | 192 | 9.54 | 66.74 | 67.44 | 67.08 |
| OTTC + 上游 jsd | MFA IPA | 158 | 11.21 | — | — | 无真实参考边界 |
| OTTC + 上游 jsd | ZIPA IPA127 | 192 | 19.91 | — | — | 无真实参考边界 |

测试仅在最终第 40 轮 checkpoint 上进行，未按测试结果选择 checkpoint。MFA 使用其中 158 条完整词典标签语音，其余三个头均评估全部 192 条。

本次 seed 42 的最终结果：原始 OTTC ARPAbet 头的 PER 为 CTC 10.26%、CE 9.91%、上游 jsd 9.68%；边界 F1 为 52.92%、63.72%、67.62%。IF-MDD ARPAbet 头的识别以 CTC 最低（9.00%）；MFA 词典 IPA 目标的 PER 以两 OTTC 组较低（CE 11.25%、jsd 11.21%，CTC 19.55%）；ZIPA token error 以 CTC 最低（9.49%，CE 12.97%，jsd 19.91%）。因此不同 head 的最佳 objective 不同；单 seed 不提供统计显著性结论。

## Forced alignment 与 text-free inference 的评估区别

Text-free inference 只输入音频，参考 phones/timestamps 仅在预测完成后用于评分；识别报告 PER 或 ZIPA token error，分段报告自由预测边界。Forced alignment 提供有序参考 phone sequence，不提供真实时间戳；它评估已知内容条件下的对齐，而不是自由识别，因此不报告 forced PER。

统一 forced_k2：三组复用 IF-MDD 的 native-k2 FSA 实现：ctc_graph、DenseFsaVec、intersect_dense_pruned、shortest_path；保留原 beam retries，关闭 torchaudio/PyTorch 对齐回退。blank_policy=previous，blank 间隙归到前一个音素；时间戳按原实现 duration / frames 转换。原生 forced_native_ot：仅两个 OTTC 组，使用音频预测的冻结 frame masses，按参考 token 数的等质量目标做 inverse-CDF 对齐，保留分数帧边界。两种 FA 算法分开报告，禁止利用真实音素持续时间或拟合时间偏移。当前 FA 输入为各头的参考音素序列，不是只给单词文本再做 G2P 的评估。

## Forced alignment：最终 test 对齐结果

| Objective | FA decoder | Head | 参考序列条数 | F1@20ms % ↑ | Start MAE ms ↓ | End MAE ms ↓ | Mean token IoU % ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CTC | forced_k2 | 原始 OTTC ARPAbet | 192 | 51.92 | 42.18 | 42.94 | 35.06 |
| CTC | forced_k2 | IF-MDD ARPAbet | 192 | 51.69 | 42.04 | 43.72 | 35.01 |
| CTC | forced_k2 | MFA IPA | 158 | 无真实参考边界 | — | — | — |
| CTC | forced_k2 | ZIPA IPA127 | 192 | 无真实参考边界 | — | — | — |
| OTTC + CE | forced_k2 | 原始 OTTC ARPAbet | 192 | 60.78 | 20.43 | 21.19 | 56.70 |
| OTTC + CE | forced_k2 | IF-MDD ARPAbet | 192 | 60.40 | 20.73 | 21.51 | 56.64 |
| OTTC + CE | forced_k2 | MFA IPA | 158 | 无真实参考边界 | — | — | — |
| OTTC + CE | forced_k2 | ZIPA IPA127 | 192 | 无真实参考边界 | — | — | — |
| OTTC + CE | forced_native_ot | 原始 OTTC ARPAbet | 192 | 58.16 | 31.79 | 32.74 | 45.49 |
| OTTC + CE | forced_native_ot | IF-MDD ARPAbet | 192 | 57.25 | 30.30 | 31.36 | 46.94 |
| OTTC + CE | forced_native_ot | MFA IPA | 158 | 无真实参考边界 | — | — | — |
| OTTC + CE | forced_native_ot | ZIPA IPA127 | 192 | 无真实参考边界 | — | — | — |
| OTTC + 上游 jsd | forced_k2 | 原始 OTTC ARPAbet | 192 | 64.78 | 18.24 | 19.00 | 59.42 |
| OTTC + 上游 jsd | forced_k2 | IF-MDD ARPAbet | 192 | 64.13 | 18.65 | 19.36 | 59.27 |
| OTTC + 上游 jsd | forced_k2 | MFA IPA | 158 | 无真实参考边界 | — | — | — |
| OTTC + 上游 jsd | forced_k2 | ZIPA IPA127 | 192 | 无真实参考边界 | — | — | — |
| OTTC + 上游 jsd | forced_native_ot | 原始 OTTC ARPAbet | 192 | 60.84 | 30.84 | 31.79 | 46.99 |
| OTTC + 上游 jsd | forced_native_ot | IF-MDD ARPAbet | 192 | 60.18 | 29.19 | 30.19 | 48.55 |
| OTTC + 上游 jsd | forced_native_ot | MFA IPA | 158 | 无真实参考边界 | — | — | — |
| OTTC + 上游 jsd | forced_native_ot | ZIPA IPA127 | 192 | 无真实参考边界 | — | — | — |

K2 phone spans 使用 IF-MDD 的 previous-blank 规则，首尾覆盖语音；native OT spans 为 transport 连续分区。两者的分段算法及帧时间定义不同，应主要在同一 decoder 内比较三组。MAE/IoU 按参考序列位置逐 token 计算；两 IPA 头只导出预测 spans，没有真实 reference intervals，不报告边界准确率。

| Objective | FA / text-free 评估 W&B |
| --- | --- |
| CTC | https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/qtheql5y |
| OTTC + CE | https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/xdpcre4z |
| OTTC + 上游 jsd | https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/vxq25ydx |

重放检查确认：显式不传 labels 的 forward 与传 labels 仅计算 loss 的 logits 完全一致；新评估的全部 text-free frame paths 与原 dev/test 预测逐帧一致。新指标存储于 evaluation_modes/{ctc,ce,jsd_upstream}/，单独 W&B evaluation run 保留原训练曲线。

## 最终 dev 指标（40 epochs）

| Objective | Head | Dev epoch | 原生识别错误率 % ↓ | 边界 F1@20ms % ↑ |
| --- | --- | --- | --- | --- |
| CTC | 原始 OTTC ARPAbet | 40.0 | 9.31 | 51.57 |
| CTC | IF-MDD ARPAbet | 40.0 | 8.06 | 51.69 |
| CTC | MFA IPA | 40.0 | 20.37 | 无真实参考边界 |
| CTC | ZIPA IPA127 | 40.0 | 8.27 | 无真实参考边界 |
| OTTC + CE | 原始 OTTC ARPAbet | 40.0 | 8.22 | 64.08 |
| OTTC + CE | IF-MDD ARPAbet | 40.0 | 8.17 | 63.02 |
| OTTC + CE | MFA IPA | 40.0 | 11.72 | 无真实参考边界 |
| OTTC + CE | ZIPA IPA127 | 40.0 | 11.29 | 无真实参考边界 |
| OTTC + 上游 jsd | 原始 OTTC ARPAbet | 40.0 | 8.34 | 68.79 |
| OTTC + 上游 jsd | IF-MDD ARPAbet | 40.0 | 8.40 | 67.63 |
| OTTC + 上游 jsd | MFA IPA | 40.0 | 12.28 | 无真实参考边界 |
| OTTC + 上游 jsd | ZIPA IPA127 | 40.0 | 18.05 | 无真实参考边界 |

两个 ARPAbet 头和 MFA 完整 phone token 报告各自标签上的 PER；MFA 指标以词典发音为目标，ZIPA 按 token error 计算，不是同粒度音素 PER。分段仅两个 ARPAbet 头有真实 PHN 时间参考；按 20ms argmax 变化（含 blank）、无拟合偏移、一对一 20ms 容差匹配。MFA/ZIPA 不伪造 token 时间边界。最终测试仅在全部 40 轮完成后进行，测试集不用于选 checkpoint。

## 并行执行记录

CTC 在 3506874.opbs / mg0022 完成。CE、JSD 首段 3506924.opbs / mg0038、mg0039（12:57–13:23）成功保存断点；第二段 3506938.opbs / mg0007、mg0030（13:23–13:36）恢复至 40 轮，两个 debug 作业 Exit_status=0。每配置独立 GPU、world_size=1，1500s runtime budget 保留原完整 scheduler，恢复沿用同一 W&B ID。三组全部完成后，已取消未启动的常规兜底 3506940.opbs / 3506941.opbs，并释放交互节点。

## 此前已完成的单头 fused-phoneme 实验

以下为此前单头模型的最终 40-epoch 测试结果，与本次四头实验采用不同训练及评估范围。README kaldi_test 为含 SA 的 240 条，core test 为 192 条。原仓库 jsd 的实际代价为 0.5 KL(m||p)+0.5 KL(m||qε)，不是标准 JSD；标准 JSD 为 0.5 KL(p||m)+0.5 KL(q||m)。

| 单头模型 | 测试集 | Raw PER % ↓ | 边界 F1@20ms % ↑ |
| --- | --- | --- | --- |
| Soft-DTW CE gamma=0.1 | readme_kaldi_test_240 | 74.61 | 13.02 |
| CTC | readme_kaldi_test_240 | 8.01 | 53.93 |
| OTTC CE | readme_kaldi_test_240 | 8.61 | 62.35 |
| OTTC upstream jsd | readme_kaldi_test_240 | 8.61 | 65.29 |
| OTTC standard JS | readme_kaldi_test_240 | 15.53 | 61.53 |

单 seed，未做学习率/gamma 调参或置信区间。Soft-DTW CE gamma=0.1、normalize=False、lr=5e-5 在当前配置中出现严重删除，不能解释为调参后的最佳性能。分段指标不等同于论文未明确实现协议的 starting-frame F1/IDR。

历史 loss 曲线报告：https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/reports/TIMIT-fused-phonemes:-loss下降与识别及分段对比--VmlldzoxODA3NDE2NA== 。五组历史曲线标记 backfill，保留原记录点，不插值；补录时间不是训练时间。

## 可追溯产物与下一步

仓库：/work/gm64/m64000/OTTC。四头配置：reproduction/multihead_timit/README.md；模型：reproduction/multihead_timit/models/{ctc,ce,jsd_upstream}/；历史报告：reproduction/RESULTS_JS_ZH.md、RESULTS_SOFTDTW_ZH.md、WANDB_ZH.md。三组已保存 final checkpoint、dev/test metrics、每组 400 条 dev / 192 条 test 逐句预测及 W&B evaluation artifact；全部 12 个识别指标已从逐句预测重新计算核验，见 validation/final_results_audit.json。下一阶段可扩到 LibriSpeech 100h，目前尚未启动。

后续 LibriSpeech 100h / 960h 与 MFA English 全套已安排，作业、资源和恢复协议见 [全套实验调度 README](../../english_suite/README.md)。正式训练是否开始以对应 PBS/W&B 状态为准。

