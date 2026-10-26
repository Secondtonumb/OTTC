# TIMIT CROTTC + 原始 JSD（AG）：四 seed

用户给出的 AG 行来自 `reproduction/timit_seed_sweep_20261022/REPORT_ZH.md`。
本目录补充 seed42/43/44/45 的匹配 CROTTC-AG，而非把旧20ms CROTTC结果并入10ms表。
此模型为单 monophone OTTC head；不是上一任务的 CTC-FiLM 双头实验。

原始 facebook/wav2vec2-large-lv60 SSL 初始化，新随机任务头；native10ms，
冻结前六层CNN，末层CNN/Transformer/phone head/learned mass head训练。
40 epoch / 2920 optimizer updates / logical batch64 / LR2e-4 / warmup500 /
weight decay0.005 / regular forward / BF16计算、FP32参数及保存权重。
train/dev/test4620/400/192，复用既有数据与音素标签。预检逐项核对四seed
AG baseline的优化器配置、可训练策略、清单及词表hash；见 `PREFLIGHT.json`。

## CROTTC 的新增部分

直接复用 `reproduction/crottc_timit_20ms_20261019/common.py` 中的 `CROTTC`、
`SpeechViews` 和 `pair_chunks`。使用冻结的 IF-MDD `_compute_cr_loss` 方法；
源码和SpeechBrain增强来源可查旧配方的 `vendor/provenance.json`。

共享一次 SpeedPerturb95/100/105，再生成两路独立 DropFreq/DropChunk；
保持两路相同时间轴和padding。保持Augmenter原默认组合语义。
CTC风格的增强是外部训练增强，验证/测试不使用双视图、不加CR。
OTTC transport 始终调用仓库共享 `ot_loss.py`，使用 `divergence='ag'`，
与历史 `jsd=True` 的原始AG分支数值一致；不是标准JSD。

```text
raw_loss = 0.5 * (AG_OTTC(view1)_sum + AG_OTTC(view2)_sum)
           + 0.1 * legacy_bidirectional_teacher_detached_KL_mean
optimizer_loss = raw_loss / number_of_source_utterances
```

保留原IF-MDD CR分母 `B*T*C - padding_frame_count`，不另改成有效元素分母。
CR的相对数值可能很小；本任务不按test调整权重。完整逻辑batch先增强一次，
在统一padding网格上按配对视图切microbatch；CR按完整batch分母汇总，
不为各microbatch分别重抽语速或重复取平均。每步记录原始OTTC/CR/加权CR/总loss。

## 验证、保存与调度

`validate.py` 在四个种子的真实增强微型模型上检查：双视图/相同时间轴、padding、
原始CR公式与teacher detach、前六层CNN冻结、末层CNN与两个head的有限非零梯度、
完整batch与microbatch的loss/梯度一致、独立epoch权重、完整Trainer断点精确恢复。
`validation/CPU_PASS.json` 为CPU结果。GPU作业还必须通过CUDA验证，之后分别运行
该seed的完整模型真实TIMIT profile，比较microbatch16/32/64、最长64句（最长语速）、
代表batch64、尾批12，检查吞吐、显存、K2 FA、原始SSL逐张量身份及FP32保存权重。
profile不会作为正式训练初始化，只有profile PASS且source hashes一致才开始训练。

每seed独立目录/模型/随机状态/W&B ID。每轮原子保存独立FP32权重并保留全部40轮，
另有best-dev及最新两个完整Trainer状态。最终也计算final5/final10的FP32权重平均，
但它们与四seed的single best-dev主表分开，候选选择仅使用dev。

每seed一张GPU，PBS每片段2小时，预留保存/评估时间；正常保存后尚未完成时
自动提交相同seed续训。失败不自动标记完成；训练和评估分别检查
`TRAIN_COMPLETE.json`、`BENCHMARK_COMPLETE.json`。
原有20ms CROTTC、已完成AG基线、FiLM任务及其source snapshot不变。

## 四seed汇总

沿用既有单best-dev的FP32评分、PanPhon PFER和冻结IF-MDD边界公式。
报告PER、PFER、无文本P/R/F1/R-value、supplied-phone FA P/R/F1/R-value/MAE。
正文表对应20ms容差；训练帧率仍为10ms。每seed从dev PER选择best-dev，
test不参与选择；四个独立模型的指标取均值和样本标准差（ddof=1）。
不把模型权重平均或平均预测当作“四seed平均”。

每次作业正常结束自动更新 `COMPARISON.json`、`PER_SEED.csv` 和 `REPORT_ZH.md`。
四seed未齐全时表中CROTTC行保持待完成，不生成貌似完整的结果。
对照AG行从四个已完成baseline的原指标/预测读取，不修改历史文件。

训练标量启用W&B；所有dev/test评估、预测和汇总留在本地，`OTTC_EVAL_UPLOAD=0`。

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" -m reproduction.crottc_ag_seed_sweep_20261022.validate --device cpu
"$LIBRI_PYTHON" -m reproduction.crottc_ag_seed_sweep_20261022.report
```
