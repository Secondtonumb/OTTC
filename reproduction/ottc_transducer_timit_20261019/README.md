# TIMIT 无 blank OTTC transducer：正式实验

用户于2026-10-19要求完整测试 TIMIT。本目录冻结已通过 GPU 短测的模型、
数据读取、原版 upstream-JSD 和 IF-MDD scorer，采用局部 import 保持训练源码稳定。
`SOURCE_SNAPSHOT.json`记录原始来源、hash及全部适配；模型/loss主体的AST与原实现一致，
没有重写 divergence。完整方法说明见[原实现](../ottc_transducer_timit_20261017/README.md)。

这不是标准 RNN-T likelihood：Encoder + Predictor + Joint 使用单调 OTTC 对齐，
分类输出仅40个音素，没有blank；独立log-count头从音频预测输出数量。重复标签按
occurrence分列，不做CTC合并。推理不使用真实音素数量或参考文本。

配置为 Wav2Vec2-large-LV60、seed42、40 epochs /2920 updates；冻结CNN、训练Transformer。
AdamW lr2e-4、weight decay.005、warmup500、线性衰减；microbatch8、累积8次，有效batch64。
主损失为原版Joint-JSD OTTC，加权重1的log-count Huber，按utterance平均。
train4620 / dev400 / core-test192，沿用既有TIMIT40 manifest及词表。

每epoch完整评测dev，按raw PER选单个best-dev；完成后先比较best-dev、final-5、final-10
的dev结果，再独立报告三者test结果。测试集不用于选择。报告raw/no-sil PER、数量MAE、
text-free分段P/R/F1/R-value@10/20/30ms；分段为预测数量下的质量分位点，非强制对齐。
现有TIMIT40清单无相邻相同标签，真实重复识别能力不能由这组数据验证；重复机制有专门合成测试。

保留全部40个独立epoch权重、best-dev、完整latest resume及平均成员路径/epoch/step/hash。
训练曲线保持W&B；依据10月19日AGENTS.md，所有测试、profile、评测指标和预测、报告仅保存本地。
W&B控制台及代码捕获已关闭，防止本地打印的评测JSON被控制台日志上传。

`run.pbs`在单卡GH200的同一次分配中先运行CUDA单元测试与4步真实数据profile
（最长音频、最多目标、普通批次、最短音频）。通过后独立初始化正式40轮训练，
profile权重不会用于正式训练。分配时限为1小时，训练预算至分配第50分钟，预留保存及评测时间。
PBS预算耗尽时保留resume；重新提交同一脚本会继续同一run和W&B ID。

```bash
qsub reproduction/ottc_transducer_timit_20261019/run.pbs
```

正式目录为`runs/jsd_blankfree_seed42`。调度提交记录见`SUBMITTED.json`；实际运行日志及GPU
遥测见`jobs/JOBID/`。`TRAIN_COMPLETE.json`表示完整训练及候选评测结束，
`REPORT_COMPLETE.json`还要求核对全部预测覆盖、40个epoch权重、平均成员和dev-only选择。
最终结果见正式目录中的`RESULTS_ZH.md`与`RESULTS.json`。提交成功不等于训练已经开始或完成。
