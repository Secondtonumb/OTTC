# TIMIT 四 phoneset 联合训练

**最终状态：CTC、OTTC CE、OTTC 上游 jsd 均已完成 40/40 epochs、2920/2920 updates；K2-FSA forced alignment 与 text-free inference 评估也已完成。**

完整的最终状态、loss 曲线、四头结果及 W&B 链接见
[最新实验报告](notion_update/README.md)。下方调度记录保留执行历史。

| 实验 | 最后记录 train loss | 状态 |
|---|---:|---|
| CTC | 0.4765 | 完成 |
| OTTC CE | 0.5157 | 完成 |
| OTTC 上游 jsd | 1.3907 | 完成 |

![三组实际记录的训练 loss](notion_update/figures/train_losses.png)

用户确认先训练 TIMIT，之后再扩到 LibriSpeech 100h。每个模型共享一个
Wav2Vec2-large-LV60 编码器，同时训练四个输出头；三个独立模型分别使用
CTC、OTTC+CE、OTTC+原始上游 `jsd`。这是一项四头扩展实验，不能作为
未经修改的上游单头 README 复现报告。之前的原始复现及 checkpoints 保留在
`reproduction/README.md` 指定的路径。

## 数据和标签

| 头 | 输出数（包含特殊标签） | 标签来源 | train / dev / test 有标签语音 |
|---|---:|---|---|
| `ottc_arpabet` | 42 | 原始 TIMIT fused mapping，保留 q、dx，相邻相同标签合并 | 4620 / 400 / 192 |
| `ifmdd_arpabet` | 43 | 旧 IF-MDD monophone 输出表；TIMIT 60→39、q 删除、dx→d | 4620 / 400 / 192 |
| `mfa_ipa` | 81 | MFA US IPA v3.1 官方词典，选择最高发音概率，完整 phone token | 3909 / 339 / 158 |
| `zipa_ipa` | 127 | ZIPA 原生 IPA127 tokens；显式 ARPAbet→IPA，复合音拆 token | 4620 / 400 / 192 |

训练集包含原始 TIMIT train 的全部 4,620 条语音（包括 SA1/SA2）。
dev=400、core test=192，均沿用旧 IF-MDD 的无时间舍入清单；三个 split
说话人互不重叠。这与上游 `kaldi_test` 的含 SA 240 条测试范围不同。
`validation/data_audit.json` 记录数量、文件 hash 和说话人审计。
IF-MDD 完整输出表仍保留 ao/zh，但本次 39-phone TIMIT 标签不会监督它们。

MFA 训练覆盖率为 84.61%。缺词典发音时，只屏蔽该语音的 MFA loss，
其他三个头继续训练；不丢弃语音。`*_excluded.jsonl` 记录这些被屏蔽的
MFA 样本。MFA 的 81 个输出包含词典中的 spn、sil 和 blank，并非
81 个独立声学音素。这里的 MFA 标签是**词典发音监督**，没有通过
MFA 声学模型选择实际发音变体，也没有 MFA 对齐时间戳。
ZIPA 标签是映射结果，未加载 ZIPA 模型或用其伪标签训练。

## 训练协议

- 初始化：官方 `facebook/wav2vec2-large-lv60` 本地原始权重；不加载用户
  Hugging Face / IF-MDD 的训练权重。编码器 key 完整匹配，并记录 checkpoint SHA256。
- seed=42，40 epochs，有效 batch=64，microbatch=16，梯度累积=4；
  每轮最后 12 条平均拆成四个 microbatch，完整保留，每轮 73 次 optimizer update。
- CNN 冻结；共享 Transformer、四个分类头共同更新；BF16 + gradient checkpointing。
- AdamW，weight decay=0.005，warmup=500 updates，linear decay；
  CTC lr=5e-5，OTTC lr=2e-4。
- 四头等权平均。OTTC：每句 transport cost 平均；CTC：PyTorch
  `reduction='mean'`（先按目标长度归一化，再按语音平均）。MFA 部分
  有标签时取有标签语音均值，全无标签时该头贡献 0。
- OTTC 每头有自己的 frame mass predictor。完成第 30 轮后快照共享
  编码器并冻结 mass heads；最后 10 轮继续更新识别编码器和分类头。
  使用显式 epoch 冻结，区别于原始代码的近似 stop-step 公式。
- 上游 `jsd` 直接复用 `ottc/ops_loss/ot_loss.py` 中原函数，
  不是标准 Jensen–Shannon divergence。三组原始 loss 数值不能直接比较。
- 每轮 dev 评估和保存，保留最近三个 checkpoint；第 40 轮最终模型评估
  core test，测试集不用于选模型。断点恢复沿用保存的 W&B run ID。

## 指标与产物

W&B 项目：<https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset>
正式 run group 为 `timit_fourphonesets_seed42`。每组独立 run，记录总 loss、
四头 loss、学习率、epoch、optimizer step、梯度范数及每轮 dev 指标。
原始 loss 与各自相对首个 loss 的曲线均记录，不把不同 loss 的绝对量级
解释为性能差异。

识别：两个 ARPAbet 头和完整 MFA IPA token 计算各自标签上的 PER；
MFA PER 反映词典目标序列的识别率。ZIPA 报告 token error，修饰符等
也作为 token，不能称为同粒度的音素 PER。不同头的原生指标没有统一 phoneset。

分段：仅两个 ARPAbet 头使用真实 TIMIT PHN 映射后的 reference intervals。
报告 20 ms 容差下 precision / recall / F1，一对一匹配边界。
预测边界来自每 20 ms frame 的 argmax 标签变化（含 blank），未拟合时间偏移。
MFA 没有参考声学边界，ZIPA 复合 token 没有独立真实边界，因此不伪造
IPA token 时间标注或报告对应 segmentation score。

每组 `models/{ctc,ce,jsd_upstream}/` 包含：

- `run_config.json`：配置、源码/模型/数据 SHA256，及 `source_snapshot/`。
- `wandb_run.json`：在线 run 链接和恢复 ID；`train.log`、TensorBoard 本地曲线。
- `checkpoint-*` / `final/`：四头模型权重、配置和可恢复状态。
- `dev_metrics.json` / `test_metrics.json`：最终 benchmark。
- `dev_predictions.jsonl` / `test_predictions.jsonl`：逐句参考、预测序列、
  frame token IDs；最终指标和预测作为 W&B evaluation artifact 上传。
- `TRAIN_COMPLETE.json`：成功结束标志。两步 smoke 带 `smoke=true`，
  不能作为正式训练完成或收敛证据。

## 验证与启动

GPU smoke `3506852.opbs` 成功：官方 LV60 三组各两步，四头反向传播、
mask/padding、缺失 MFA 标签、dev 评估、保存 checkpoint、在线 W&B 均通过。
GPU sparse transport objective 与原上游 batched loss 数值一致。
CPU 验证了冻结快照保存/恢复，以及 MPI rank 环境清理后 world_size=1。
冻结后的模型和 optimizer/RNG 断点恢复验证通过；使用局部兼容加载支持
PyTorch 2.6 恢复本 trainer 保存的 NumPy RNG 状态。

调度遵循 [Miyabi dispatch skill](/home/m64000/.codex/skills/miyabi-dispatch/SKILL.md)：
三个独立配置、三节点、每节点一个 GPU，不在配置之间同步梯度。
rank 子进程清除 MPI/PMI distributed 标志，Python 入口断言 world_size=1。
队列 `regular-g`，walltime=4h；重提交自动续训。
正式三配置作业 `3506873.opbs` 曾提交常规队列，等待期间申请到交互节点，
因此已在该作业启动前取消，避免重复训练。
实际训练使用 `3506874.opbs`，mg0022，2h：
`train_interactive.sh` 在一个 GPU 上依次完成三个配置，模型和训练参数相同。
驱动日志 `train_interactive.log`，进程 PID 3538479。
若交互 allocation 在全部完成前结束，依赖其结束的常规续训作业
`3506904.opbs` 将恢复尚未完成的配置。已完成的配置自动跳过，
三个完成时仅汇总结果；不会重新训练或更换 W&B ID。
第一轮 CTC 的完整 400 条 dev 评估及 checkpoint-73 已保存，
在线 run：<https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/7tp3slt5>。

### 2026-10-08 并行调度更新

用户要求并行并优先使用空闲 debug 队列。CTC 保持在原交互节点运行；
CE、upstream JSD 已通过 `train_parallel_debug.pbs` 在两个独立 debug
节点启动：`3506924.opbs`，mg0038 / mg0039，30min。
每节点一模型、一 GPU、world_size=1，训练配置与原串行方案一致。
旧串行驱动跳过外派的 CE / JSD，只完成 CTC 后释放节点。

debug 的 train-loop runtime budget 为 1500s，预留初始化、checkpoint
保存和 W&B 同步时间。若达不到 40 轮，写 `CHECKPOINT_PAUSED.json`，
保持原 40-epoch scheduler，保存 optimizer/RNG；不运行最终测试或写
`TRAIN_COMPLETE.json`。续训在同一 W&B run 继续，不改变训练 epoch 数。
局部 CPU 实验已验证中途保存与恢复直至全部 epochs 完成。

第二段 debug 续训作业 `3506938.opbs` 依赖 `3506924.opbs` 结束，
继续使用两节点并行；已经完成的目标直接跳过。
常规兜底作业 `3506940.opbs`（CE）、`3506941.opbs`（JSD）依赖
第二段 debug 结束，仅恢复未完成的目标。已取消被替代的排队续训作业。
两段 debug 均保留最多 30min 的系统时限，原 40-epoch 训练计划不变。

实时 W&B：

- CTC：<https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/7tp3slt5>
- CE：<https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/ql9zhyav>
- upstream JSD：<https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/q0zlpgjh>

复用同一个单配置入口提交新的并行作业：

```bash
qsub -N timit_fourhead_ce -v OBJECTIVE=ce \
  reproduction/multihead_timit/train_objective.pbs
```

默认 regular-g / 1node / 2h；W&B 在线、effective batch 64、microbatch 16，
每 epoch 评估和 checkpoint；真实 reference segmentation 仅两个 ARPAbet 头。

## 最终训练状态及双模式评估

三组已完成 40 epochs / 2920 updates，`final/`、400 条 dev / 192 条 test
逐句预测和 W&B evaluation artifacts 全部保存。全部 12 个识别错误率已
从逐句预测重新计算核验，见 `validation/final_results_audit.json`；
W&B 三个训练 run 均为 finished，云端指标/artifact 核验见
`validation/final_wandb_audit.json`。闲置交互 allocation 与未启动常规兜底
已释放/取消，旧作业记录为执行历史。

原测试结果对应 **text-free inference**：模型只输入音频，greedy argmax
按 CTC-style collapse 输出序列；真实 phones/时间仅用于评分。
报告各 head 原生 PER/token error，两个 ARPAbet 头报告自由预测边界。

用户追加要求将测试分成 forced alignment 与 text-free inference。
`evaluate_multihead_modes.py` 使用同一最终 checkpoint，对 dev/test 补充：

- `text_free`：显式 forward 不传 labels；逐帧核对原预测完全一致，
  并验证计算 loss 时传 labels 不改变 logits。
- `forced_k2`：三组共同复用 IF-MDD 的 native-k2 FSA graph、DenseFsaVec、
  intersect_dense_pruned、shortest_path 与原 beam retries；禁止 torchaudio
  fallback。参考序列提供给解码器，不提供真实边界。
- `forced_native_ot`：两个 OTTC 组的原生 frame-mass transport，按
  target token 等质量和 inverse CDF 导出 fractional-frame 连续分区。

FA 不报告 PER，因为标签已提供。有真实参考时间的两个 ARPAbet 头报告
10/20/30ms 边界 P/R/F1、逐 token start/end MAE、hit@20ms、mean IoU；
MFA/ZIPA 仅导出 spans，不伪造独立 token 时间真值。
固定 phone sequence FA 不等同于仅提供词级 transcript 再做 G2P。
K2 采用原 IF-MDD 的 blank_policy=previous、duration / frames 时间转换，
native OT 为 inverse-CDF 连续分区；跨 decoder 比较还受算法和帧时间定义影响。

用户明确要求 K2-FSA 后，旧 Viterbi 评估 `3507451.opbs` 已取消；
已产生的结果移至 `superseded_evaluations/`，不列为当前 FA 结果。
K2-FSA 评估作业 `3507459.opbs` 暴露 K2 Fsa 与 PyTorch inference_mode
的 tensor version-counter 兼容问题；仅在 K2 调用处关闭 inference_mode 并
复制 normal tensors 后重提交 `3507467.opbs`，debug-g / 1node / 20min，
无需重新训练，模型 inference 仍保持 no-grad。
精确复用函数及 SHA256 记录在 `ifmdd_k2_source_snapshot.py` 和 metrics.json。
评估保存 `evaluation_modes/{objective}/`，每 objective 独立 W&B evaluation run，
原始训练 history 保留。汇总内容随 `prepare_notion_update.py` 更新到
`notion_update/README.md`。

三组 K2 双模式评估已完成（`3507467.opbs`，mg0034），dev=400 / test=192。
所有四头对齐均成功，MFA 158 条 test 有参考序列，其余 192 条；
不传 labels 的 logits 与传 labels 仅计算 loss 的 logits 完全一致，
全部 text-free paths 与训练结束的原预测逐帧相同。
核验文件：`validation/k2_modes_audit.json`；W&B 云端 summary / artifacts
核验：`validation/k2_modes_wandb_audit.json`。

原始 OTTC ARPAbet 头的最终 core test：

| Objective | Text-free PER % ↓ | Text-free F1@20ms % ↑ | K2 FA F1@20ms % ↑ | K2 start / end MAE ms ↓ |
|---|---:|---:|---:|---|
| CTC | 10.26 | 52.92 | 51.92 | 42.18 / 42.94 |
| OTTC CE | 9.91 | 63.72 | 60.78 | 20.43 / 21.19 |
| OTTC upstream jsd | 9.68 | 67.62 | 64.78 | 18.24 / 19.00 |

FA 标签已给定，不报告识别 PER。FA K2 和 text-free argmax 的边界提取
及时间转换不同，不能把两列 F1 相减当作固定内容带来的改善幅度。
所有四头及原生 OT 的完整对齐结果列在 `notion_update/README.md`。

```bash
qsub -o /work/gm64/m64000/OTTC/reproduction/multihead_timit/train_job.log \
  reproduction/multihead_timit/train.pbs
```
