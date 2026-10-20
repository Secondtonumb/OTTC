# 本次真实 MFA 对齐（四条可视化示例）

已实际完成四条语音的词转写强制对齐。工具 MFA 2.2.17，官方 english_us_arpa 声学模型及发音词典 v3.0.0，未改动模型或词典。不是论文 MFA 3.0 基线复现，也不是完整测试集评估。

|语料|ID|输出音素区间数|
|---|---|---:|
|TIMIT|FDHC0_SI1559|49|
|Buckeye|s2101a_10|49|
|Libri clean|1089-134686-0001|32|
|Libri other|1688-142285-0002|26|

四条输入均为16kHz单声道PCM WAV及原始词转写，词典OOV为0；未使用参考音素或参考时间。输入来源和SHA见INPUT_MANIFEST.json；原始输出位于textgrids/，数值导出alignments.jsonl保留原始带重音标签，显示标签去重音、ao→aa、zh→sh。

在OTTC根目录执行 `bash reproduction/mfa_direct_20261017/run_align.sh` 使用本目录隔离ARM运行环境，通过官方PretrainedAligner的align/analyze_alignments/export_files实际运行Kaldi；原始完成日志align_api_native.log，耗时约52秒。输入已是原生WAV，无需重采样；采用SQLite。现有MFA3环境缺少ARM Kalpy，本次使用不依赖Kalpy的2.2.17，未把工具版本与模型版本混称。依赖精确列表在requirements_frozen.txt及conda_packages.json。

模型及词典来自[官方MFA模型发布仓库](https://github.com/MontrealCorpusTools/mfa-models/releases)，下载URL、大小、SHA见models/DOWNLOAD_MANIFEST.json。完整来源和冻结评分代码hash见RESULTS_COMPLETE.json。

图中10/20/25ms的P/R/F1/R-value采用可视化统一参考及一对一边界匹配，保存于FIGURE_BOUNDARY_METRICS.json，全部为强制对齐结果。TIMIT另以历史冻结IF-MDD raw-PHN scorer复算（dx→d，闭塞音→静音，评分时过滤静音，不对时间取整），结果单独保存在RESULTS_COMPLETE.json；本例音素数不等，不报告TSE，也未执行共同等音素数子集比较。二者参考和协议不同，不混用分数。

[更新后的交互页面](../segmentation_figures_20261017/interactive/index.html)支持点击区间高亮、只播放区间并在终点暂停；所有图和TextGrid已重新导出。
