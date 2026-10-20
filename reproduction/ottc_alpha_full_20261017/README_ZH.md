# 全量 α 解码评估

对三个既有的原始 upstream-JSD 单头 Wav2Vec2-large 模型做音频推理，比较 argmax、α 取整到帧、α 帧内插值。固定测试集为 TIMIT core-test 192、Buckeye test 1,840、Libri test-clean 2,620 和 test-other 2,938，共 7,590 条；不含开发集、跨语料 Buckeye 或其他模型。

[全量数值报告](RESULTS_ZH.md)、[机器可读结果](RESULTS.json)、[完整 P/R/F1/R-value 表](BOUNDARY_METRICS.csv)、[对比图](full_test_boundary_f1.png)、[事后诊断](DIAGNOSTICS_ZH.md)、[W&B run 记录](WANDB_RUNS.json)。

`INPUTS.json` 只包含音频输入、模型身份和用于事后回放校验的历史音频识别结果。GPU 解码不读取 `SCORING_REFERENCES.json`。每条语音重新计算 α，以自身 greedy 识别音素数 Û 设置均匀 β=1/Û；无参考序列、数量或时间输入。参考只用于 CPU 评分，静音/blank 策略不调整。

三个方法使用相同固定 20ms 帧时钟。原 argmax 的识别序列、逐帧路径（TIMIT 为全部非 blank spans）及全量评分计数必须与既有结果一致；不能通过丢弃失败语音完成评估。每个模型先跑最长、最短和中位长度输入，再继续全部语音，进度与显存记录在 `job.log`、`PROGRESS.json`、`telemetry.csv`。每条 α 的累计质量交点均做校验，三条尺寸探针另检查运输矩阵双边边缘质量。

复现：

```bash
python -m reproduction.ottc_alpha_full_20261017.prepare
# 仅在 PBS 分配的 GPU 节点执行推理：
bash reproduction/ottc_alpha_full_20261017/run.sh
# 以下仅需 CPU，使用含 numpy、soundfile、matplotlib、wandb 的环境：
python -m reproduction.ottc_alpha_full_20261017.score
python -m reproduction.ottc_alpha_full_20261017.diagnose
python -m reproduction.ottc_alpha_full_20261017.plot
python -m reproduction.ottc_alpha_full_20261017.publish
```

解码结果逐条追加，各模型完成后单独写 marker。重启复用已完成记录，校验冻结输入、源码与结果 hash。全部 7,590 条覆盖且 checkpoint 推理前后 hash 一致后才写 `DECODE_COMPLETE.json`；全量评分与历史基线复现检查通过后写 `SCORE_COMPLETE.json`。`INTERIM*` 仅为过程中已完整完成的模型快照，不代表最终结果。

W&B 只记录数字摘要、协议与 checkpoint hash，三个模型各一个独立 run，保留 run ID；不上传音频、文本或逐条附件。此任务无训练，未计算 loss，不填造训练曲线或零 loss。四条可视化样例与其导出仍在原目录，本目录存完整测试集结果。
