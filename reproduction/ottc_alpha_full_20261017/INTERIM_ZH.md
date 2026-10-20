# upstream-JSD 全测试集 α 解码对比

状态：partial；已覆盖 2,032 条语音。使用三个既有单头模型，不重训，不更新参数。

三种解码均为无文本分段，识别序列与历史结果逐条一致。α 的目标质量为 1/自身预测音素数，不使用参考音素数、序列或时间。

下表为边界 F1 / R-value（%，20ms容差）；按全测试集累计匹配数评分。

|测试集|条数|argmax|α取整到帧|α帧内插值|无sil PER %|
|---|---:|---:|---:|---:|---:|
|timit_core_test_192|192|65.36 / 70.25|60.81 / 66.32|61.12 / 66.60|9.645|
|buckeye_test|1840|54.06 / 58.97|50.68 / 56.00|50.33 / 55.67|19.635|

P/R/F1/R-value 在 10/20/25/30ms 的完整结果见 BOUNDARY_METRICS.csv。原 argmax 的匹配数、边界数、PER 已与历史全量结果核对；推理前后 checkpoint hash 不变。

argmax→α取整衡量按累计质量重新分段的作用；α取整→α插值衡量分数帧时间戳的作用。20ms 为事先沿用的主指标，未根据测试结果调时间偏移或挑容差。

TIMIT/Buckeye 使用人工音素边界；Libri 使用既有自动对齐，模型身份未经确认，不应解释为人工标注精度。α 插值是假定一帧内质量均匀，并非新增声学时间分辨率。

本方法保留原识别序列。若缺少 sil，不会额外推断静音路径；归一化 α 总和为1，本身不能确定音素数。

复现：GPU 执行 `bash reproduction/ottc_alpha_full_20261017/run.sh`；CPU 执行 `python -m reproduction.ottc_alpha_full_20261017.score`。

逐条原始 α 和边界位于各模型的 *_predictions.jsonl；参考仅在 SCORING_REFERENCES.json。源文件、输入与权重 SHA256 见 DECODE_CONTRACT.json、RESULTS.json 和各 *_COMPLETE.json。
