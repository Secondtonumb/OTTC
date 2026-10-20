# Libri100 单头：已完成整套 split 的 K2-FSA FA

阶段性报告：只评分已覆盖全部参考句子的 split，不把正在写入的部分集合当作完整测试。三组最终总报告另见 REPORT_ZH.md。

给定真实音素序列，无参考时间戳；20ms 模型。全集一对一边界 F1/R，MAE 使用同一 split 当前已完成模型共同成功对齐的句子。Libri参考为已有自动对齐，Buckeye为人工标注零样本测试。

| Split | OTTC代价 | K2 FA F1@20ms ↑ | R-value ↑ | 共同子集边界 MAE ↓ | 对齐失败 |
|---|---|---:|---:|---:|---:|
| dev_clean | CE | 61.74% | 67.65% | 34.68 ms | 0/2703 |
| dev_clean | standard JSD | 62.92% | 68.63% | 69.78 ms | 0/2703 |
| dev_other | CE | 58.22% | 64.78% | 54.39 ms | 1/2864 |
| dev_other | standard JSD | 59.06% | 65.47% | 82.33 ms | 1/2864 |
| test_clean | standard JSD | 61.89% | 67.78% | 84.26 ms | 0/2620 |
| test_other | standard JSD | 58.94% | 65.38% | 88.35 ms | 0/2938 |
| buckeye_dev | standard JSD | 54.31% | 61.22% | 30.01 ms | 0/1524 |
| buckeye_test | standard JSD | 49.61% | 57.28% | 35.44 ms | 0/1840 |

每个 split 当前完成的模型及共同子集规模、P/R、10/20/30ms 指标和路径审计结果保存在 FA_COMPLETED_SPLITS.json。所有 split 推理完成后，原评测流程统一上传完整指标至 W&B。

Libri dev-other 的一条仅sil参考记为FA不适用，保留全集统计。standard JSD全套推理已完成；原生最终指标与此处重评分一致。
