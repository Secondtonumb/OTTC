# CIF 与纯 upstream-JSD OTTC：英语语料统一口径比较

仅重评分保存的预测。参考为纯 OTTC 的 speech-phone 区间，去 sil/q、dx→d；边界为区间端点去重、句首句尾排除20ms、一对一匹配、20ms容差，固定20ms编码器帧时钟。CIF保留分数帧crossing，不拟合偏移。R-value复用冻结IF-MDD公式。

全部是无文本推理。纯OTTC固定第40轮；CIF按dev无文本分段R-value选单个checkpoint。相同预训练Wav2Vec2-large-LV60、相同训练语料，但冻结日程、训练标签重复/静音处理及decoder不同，不是单因素消融。Libri参考为身份未核实的自动对齐，Buckeye为人工边界。

|训练语料|测试集|方法|PER↓|P|Recall|F1|R-value↑|
|---|---|---|---:|---:|---:|---:|---:|
|buckeye|test|native_ottc|19.635|51.587|56.788|54.063|58.971|
|buckeye|test|ottc_jsd_0p5|20.793|54.306|54.333|54.320|61.003|
|buckeye|test|ce|20.225|52.991|51.246|52.104|59.494|
|buckeye|test|ctc_0p5|20.632|54.840|54.997|54.919|61.486|
|libri100|test_clean|native_ottc|3.353|77.398|75.388|76.380|79.842|
|libri100|test_clean|ottc_jsd_0p5|4.969|55.528|53.245|54.363|61.455|
|libri100|test_clean|ce|6.242|54.424|53.297|53.855|60.846|
|libri100|test_clean|ctc_0p5|6.115|53.263|51.375|52.302|59.685|
|libri100|test_other|native_ottc|6.223|74.423|72.386|73.390|77.330|
|libri100|test_other|ottc_jsd_0p5|8.173|55.487|52.773|54.096|61.292|
|libri100|test_other|ce|8.815|55.306|53.916|54.602|61.519|
|libri100|test_other|ctc_0p5|9.911|53.917|51.488|52.674|60.081|
|libri100|buckeye_test|native_ottc|31.148|54.394|59.318|56.749|61.577|
|libri100|buckeye_test|ottc_jsd_0p5|30.652|49.885|53.046|51.417|57.469|
|libri100|buckeye_test|ce|32.470|50.058|54.516|52.192|57.596|
|libri100|buckeye_test|ctc_0p5|32.534|50.265|52.206|51.217|57.762|

纯OTTC的原始PER和边界计数全部精确复算通过。输入文件hash、逐组参考标签差异句数及完整计数见 english_metrics.json。

复现：python -m reproduction.cif_comparison_20261016.compare_english。

额外核验：Libri test-clean 有719句、test-other有792句，两套原始评分参考序列仅在相邻重复音素合并上不同；折叠相邻重复后序列全部一致。本报告一律使用纯OTTC未合并的 speech-phone 参考，因此CIF PER与此前原报告有差别。

本次观察：Buckeye训练的CIF+OTTC相对纯OTTC，test PER差1.158个百分点、F1高0.257、R-value高2.032；Libri训练模型在Buckeye零样本test，CIF+OTTC PER好0.495个百分点、F1低5.333、R-value低4.108。单seed且训练配方不同，未验证统计显著性。
