# CIF TIMIT40 完成结果

三组均训练40 epoch/2920更新；按最大dev text-free R-value@20ms选择checkpoint，测试集不参与选择。无文本分段，冻结IF-MDD directional greedy，fractional duration/T时钟，20ms容差。

|实验|split|选择epoch|PER raw %|PER nosil %|P %|Recall %|F1 %|R-value %|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|ottc_jsd_0p5|dev|11|13.653|13.927|57.137|56.580|56.857|63.278|
|ottc_jsd_0p5|test|11|15.383|15.861|57.513|57.077|57.294|63.628|
|no_ctc|dev|10|13.883|13.961|57.245|58.604|57.917|63.793|
|no_ctc|test|10|14.548|14.557|59.117|60.787|59.940|65.480|
|ctc_0p5|dev|24|9.312|8.770|58.853|57.824|58.334|64.601|
|ctc_0p5|test|24|10.305|10.139|58.661|57.737|58.195|64.468|

每组dev400/test192预测ID数量、唯一性已核对。frame-grid指标、原始路径和文件hash见状态快照：`reproduction/cif_timit_20261016/status/track_20261016_175454.json`。

本次单seed结果：test分段无辅助组最高；CTC组识别最好。dev最优分段点和40轮末点不是同一checkpoint，不应混报。与旧OTTC/CE历史评分的时钟和边界协议不同，不直接对比数值。

Buckeye三组最长音频profile通过；正式训练作业已解除依赖，目前排队，尚无正式训练history/checkpoint。
