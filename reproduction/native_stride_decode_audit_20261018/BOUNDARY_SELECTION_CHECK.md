# 按边界指标重新选择权重的核查

复用已完成的独立FP32预测，不重新训练或推理。TIMIT dev400选择、core-test192报告。每项仅在原best-dev/final/final5/final10四候选中选择，test不参与。

非FA按dev固定10ms时间轴的20ms边界R-value选择；FA按dev的20ms边界R-value选择。另单列按dev MAE选择的FA，不能混用选择标准。

| OTTC代价 | 非FA单音素 F1/R | 非FA S3T2** F1/R | FA单音素 F1/R | FA S3T2** F1/R |
|---|---:|---:|---:|---:|
| jsd_upstream | 58.31/64.72 | 51.03/58.89 | 56.07/62.69 | 49.60/57.22 |
| ce | 57.42/64.12 | 48.57/56.94 | 54.70/61.53 | 48.66/56.42 |
| jsd_standard | 56.55/63.45 | 47.13/56.20 | 50.10/57.64 | 48.25/56.07 |

FA按dev MAE选择（越低越好）：

| 目标 | 单音素所选 | 单音素 test MAE/ms | S3T2**所选 | S3T2** test MAE/ms |
|---|---|---:|---|---:|
| jsd_upstream | final_10 | 27.79 | final_10 | 40.49 |
| ce | best_dev | 29.12 | final_10 | 42.50 |
| jsd_standard | final_10 | 31.52 | final_10 | 44.23 |

在这些候选和这192句上，按dev边界指标重新选择后，20ms边界F1/R和FA MAE依然弱于单音素。该结论限于当前配方与候选集，不能推断topology设计普遍无效。

“所有指标、所有容差都更差”不成立：CE/standard JSD识别PER改善；原识别选择下standard JSD的FA@10ms F1由27.97到28.10，R-value由38.92到39.03，为很小的点估计提升。

底层全部候选数据与选择依据见BOUNDARY_SELECTION_CHECK.json；原按识别PER选择的结果见REPORT_ZH.md。
