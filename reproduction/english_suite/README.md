# LibriSpeech 100h / 960h 与 MFA English 全套实验

**当前重点：[OTTC 上游 JSD 的结果、checkpoint 平均及 CE 对照](JSD_RESULTS.md)。**

[Zapformer JSD 支持与验证](mfa_jsd/README.md)：直接调用已有 OTTC 复现归档，GPU loss/梯度及完整模型检查已通过；JSD 长训尚未提交。

[论文 MFA 3.0 与归档 CTM 身份核对](MFA_TOPLINE.md)：论文 test TSE=31.95ms、R-value=87.44%；归档 TIMIT_MFA CTM 与论文 Kaldi 行数值一致，系统身份仍待核实，不能称为原生 MFA topline。MFA 标签 Zapformer 另列。

[OTTC CE/JSD 与 MFA 统一评分口径](mfa_protocol_comparison/README.md)：冻结 MFA scorer、raw PHN、完整 test192 边界计分，以及各系统共同等长子集的 TSE 对照。

[✅ / 🟡 / ⏳ 已跑实验标记](EXPERIMENT_STATUS.md) · [前端差异](FRONTENDS.md)。

CE 复用规则：同一数据集与模型配置下，已完成的 CE 直接复用现有 checkpoint，只补测试和权重平均；尚未完成的从断点续训。TIMIT 四头 CE 已完成，Libri100 四头 CE 是原进程继续训练。原始单头、四头、旧 IF-MDD WavLM 与 MFA 标签 Zapformer 单独匹配，已有同名数据集不代表权重可互换。其他既有训练和评估队列按原范围保留。

[最新进度、启动时间与 W&B 验证链接](TRACKING.md)。
[全部当前结果与测试覆盖](RESULTS.md)。
[W&B dev PER 与 segmentation 曲线](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/reports/English-phonemes:-dev-PER-and-segmentation--VmlldzoxODA3OTU2MQ==)。

[三个 checkpoint：逐个测试、最后三个权重平均、dev 最好三个权重平均](../checkpoint_comparison/README.md)。
[GPU 吞吐实测与配置调整](THROUGHPUT.md)：Libri960 下一进程 microbatch=8；MFA 下一片段放宽 microbatch 限额。

2026-10-09 11:39 JST：Libri100 CTC / CE / 上游 JSD 均完成40轮及最终 text-free test，Libri960 三组均已启动。MFA CTC / OTTC / SPOTTC 已保存并排好续训，FiLM 正式训练排队。TIMIT 三组 checkpoint 比较已完成；MFA SPOTTC 的计算和 W&B 上传已完成，作业因 Markdown 汇总错误 exit1，汇总现已修复；CTC / OTTC 比较继续运行。动态状态见 EXPERIMENT_STATUS.md / TRACKING.md。
排队、运行和完成是不同状态；最新 PBS 状态见 `submission_status.json`，最终完成以模型目录的标记和 epoch 为准。

TIMIT 三组已完成 40/40 epochs、2920/2920 updates，K2-FSA 与 text-free 评估已完成。
[TIMIT 完整报告与 loss 曲线](../multihead_timit/notion_update/README.md)。

## LibriSpeech：四头三组，两个规模

| 数据 | 训练范围 | 模型 | 训练预算 |
|---|---|---|---|
| LibriSpeech 100h | train-clean-100，28,538 条、100.586h | CTC / OTTC CE / OTTC 上游 jsd | 各 40 epochs |
| LibriSpeech 960h | train-clean-100 + train-clean-360 + train-other-500，281,191 条 | 同上 | 各 40 epochs |

每个模型共享官方 `facebook/wav2vec2-large-lv60` 编码器，四头为原始 OTTC fused ARPAbet、
IF-MDD ARPAbet、MFA US IPA 完整 phone、ZIPA 原生 IPA127。两个规模分别从官方初始化训练，
不从 TIMIT 或 100h 模型热启动。seed=42；CTC LR=5e-5，OTTC LR=2e-4；CNN 冻结；
BF16、gradient checkpointing、effective batch=64；Libri100 当前 microbatch=4，
Libri960 在下一进程按实测策略使用 microbatch=8；四头等权；
OTTC 在完成第 30 轮后冻结对齐分支。上游 jsd 保留原始实现。

dev-clean + dev-other 共 5,567 条；test-clean + test-other 共 5,558 条，
训练/dev/test 说话人不交叠。保留超过 20 秒的样本；数据准备检查音频路径、词表、
参考序列与特征帧容量。MFA 词典不完整仅屏蔽对应 head，其他三个 head 继续监督。
ZIPA 使用 IPA-Pack++ 的 `supervision.text` 和原生 SentencePiece，MFA 使用最高概率词典发音。
LibriSpeech 的已有时间参考来自自动对齐，不能当作 TIMIT 人工 PHN。
该配置是四头音素实验，区别于上游 README 的 LibriSpeech 字符 ASR。

| 作业 | Job ID | 资源与用途 |
|---|---|---|
| 960h 数据准备及两规模审计 | 3507715.opbs | debug-g，1 节点，30min |
| 三目标官方 LV60 GPU smoke | 3507718.opbs | debug-g，1 节点，30min；依赖数据准备成功 |
| 正式第一段 | 3507745.opbs | regular-g，3 节点，24h；依赖 GPU smoke 成功 |
| 第二段及后续续训 | 3507769.opbs | regular-g，3 节点，24h；依赖第一段成功 |
| 首段后完整模型 K2 / text-free 评估 | 3507759.opbs | regular-g，1 节点，12h |

模式 C：每 rank 一个独立目标、每节点一个 GPU，world_size=1；各目标内部先 100h 再 960h。
每次 allocation 预留约 1 小时保存和最终评估，限时保存 optimizer/RNG，并保持完整 40-epoch scheduler。
第二段成功退出后，若仍未完成，会通过 `finish_segment.py` 提交下一段；失败退出不自动反复重试。
完成的模型跳过，续训沿用 W&B run ID。正式时长尚未实测，排队时间和训练时间分开记录。

输出：`../multihead_{libri100,libri960}/models/{ctc,ce,jsd_upstream}/`。
W&B 项目：[ottc-english-multiphoneset](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset)。
正式 group：`libri100_fourphonesets_seed42` / `libri960_fourphonesets_seed42`。
各组启动后将自己的 URL 写入 `wandb_run.json`。

## MFA English：沿用旧全量数据和原配置

用户明确选择沿用旧数据与配置。来源为
`/work/gm64/m64000/zipa/mfa_miyabi_20260922`；
数据为 `/work/gm64/m64000/dataset/english-mfa31-10ms-all-train-v2/training.json`：
1,798,637 条、2888.025 小时、10 类语料，10ms / 80 维特征。
该系列使用原 Zapformer、MFA 词表和原损失，独立于 LibriSpeech Wav2Vec2 四头系列。

| 组 | 原始恢复步数 | 目标 epochs | LR | 每组 GPU | 正式 Job ID |
|---|---:|---:|---:|---:|---|
| CTC | 5253 | 10 | 0.02 | 8 | 3507774.opbs |
| OTTC（原 CE） | 5253 | 10 | 0.02 | 8 | 3507775.opbs |
| SPOTTC | 3444 | 10 | 0.02 | 8 | 3507776.opbs |
| FiLM K4 | 从头 | 10 | 0.02 | 8 | 3512424.opbs |

模式 B：每组 true DDP，8 节点、每节点 1 GPU，regular-g，8h 限时片段。
保持旧 batch-duration 315/315/320/200、原 sampler、optimizer、RNG 和 migration contract。
四组均依赖 `3507763.opbs`（8 节点、4h）验证成功：每组真实数据更新 30 步，
并复核 CTC / FiLM 的保存后恢复。该验证没有最终 PER 或收敛结论。

旧 `runtime.sha256` 中两个调度控制脚本后来发生修改，原训练源码仍匹配。
本次使用独立快照 `mfa_review_gate.py` 与 `mfa_runtime.sha256`，
不修改旧源码、数据和 checkpoint。W&B 通过 `mfa_tracked.py` 在 AST 中仅插入
日志/进度记录调用与外部 dev hook；W&B 初始化和 dev 评估后恢复 Python、NumPy、Torch CPU/CUDA RNG。
下一续训片段由独立 `mfa_optimized.py` 调整 microbatch 切分预算，具体因子、
同模型历史测试证据和数值差异说明见 [THROUGHPUT.md](THROUGHPUT.md)。
保留原 logical batch、sampler、optimizer 与 loss 定义，变化写入本地 provenance 和 W&B。
新输出为 `mfa_models/{ctc,ottc,spottc,film_k4}/`，不覆盖旧实验。
W&B 项目：[ottc-mfa-english](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english)。

一个 8h 片段退出不表示完成 10 epochs。读取 `progress.json` 与 checkpoint progress；
正常退出后自动检查 checkpoint，未达到 10 epochs 时排下一个 8h 续训片段；
异常退出不自动反复重试。同脚本再次提交会恢复新输出的最新 checkpoint。
原全量视图含旧 LibriSpeech/L2-ARCTIC
dev/test，不能将这些样本列作未见测试；沿用原外部 TIMIT 协议，不用测试集选 checkpoint。
已补外部 TIMIT 特征准备、目标独立的 dev/test 推理及 K2 FA：
原 development 4096 条、core test 192 条；当前固定 checkpoints 先补诊断评测，
下一续训进程恢复后和每个完成 epoch 均记录 dev PER 与边界 Precision/Recall/F1。
最终 epoch10 checkpoint 自动提交完整 dev/test 与 K2 FA；不使用 test 选 checkpoint。
SPOTTC/FiLM text-free 复用原 S3T2 greedy complete；跨目标统一 K2 FA 使用 phone posterior，
三状态模型先边缘化 substate。这与原 topology 约束的 substate FA 分开说明。

## 识别与分段评估

LibriSpeech 的 `evaluate.pbs` 分别运行 audio-only text-free 和
IF-MDD 的原 K2-FSA forced alignment，另列 OTTC 原生 transport 对齐。
FA 提供参考 phone 序列，识别 PER 只用于 text-free；四头维持各自原生标签定义。
MFA/ZIPA 没有真实声学区间时，仅导出 spans，不报告伪造的 segmentation 准确率。
K2 对齐不回退到 torchaudio / 自写 PyTorch 对齐器。

## 复用命令

```bash
# LibriSpeech：每目标独立训练，三个 GPU 并行；有 checkpoint 自动续训。
qsub -v CHAIN_CONTINUE=1 reproduction/english_suite/train.pbs

# MFA：原 logical batch 的 8-GPU DDP，应用实测 microbatch 限额，沿用 GPU gate。
qsub -v ARM=ottc reproduction/english_suite/mfa_train_optimized.pbs
```

当前作业已提交，无需重复运行上述命令。实际调度系统会把合适的 regular-g 作业
归到 small-g 子队列；以 PBS 返回的 job ID、状态和开始时间为准。
