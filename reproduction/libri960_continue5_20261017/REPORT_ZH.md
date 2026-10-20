# Libri100 epoch40 → Libri960 追加5轮：三种OTTC代价

状态：全部完成。

三组分别从自己的Libri100 epoch40/17840step完整权重继承AdamW moments；新的Libri960阶段为281,191句、960.87小时、5epoch/21,970step，累计39,810 optimizer step。BF16、全局batch64、lr2e-5、warmup500、线性调度；保持既有regfrozen对齐分支及CNN冻结，Transformer继续微调。未更改任何OTTC代价公式。

每epoch在约1/3、2/3和结束保存完整可恢复checkpoint；所有5个完整epoch权重独立保留。比较固定最后一轮、按dev PER选择的best-dev（含原始epoch40基线）以及五个新epoch的平均。仅新增5轮，不能称作last10平均。所有父checkpoint和旧source snapshot保持原样。

测试从未参与模型/平均成员选择。最终变体按FP32 dev-clean/other参考音素数加权PER选择。Buckeye始终为零样本人工标注评测；Libri边界仅为既有自动对齐参考，来源模型身份未核实。

Text-free与给定音素K2 FA分开。唯一端点、去句首句尾20ms、一对一10/20/30ms时间匹配；冻结IF-MDD R-value公式、FA duration/frame_count时间网格、blank归前一音素。F1/R覆盖全集，FA失败零命中；MAE为同一目标各变体和加训前的共同成功子集。

| Objective | Variant | Split | PER ↓ | Text-free F1@20ms ↑ | Text-free R-value ↑ |
|---|---|---|---:|---:|---:|
| jsd_upstream | libri100_epoch40 | dev_clean | 3.23% | 77.57% | 80.84% |
| jsd_upstream | libri100_epoch40 | dev_other | 6.88% | 73.39% | 77.33% |
| jsd_upstream | libri100_epoch40 | test_clean | 3.35% | 76.38% | 79.84% |
| jsd_upstream | libri100_epoch40 | test_other | 6.22% | 73.39% | 77.33% |
| jsd_upstream | libri100_epoch40 | buckeye_dev | 26.79% | 61.03% | 65.81% |
| jsd_upstream | libri100_epoch40 | buckeye_test | 31.15% | 56.75% | 61.58% |
| jsd_upstream | final | dev_clean | 3.37% | 77.06% | 80.42% |
| jsd_upstream | final | dev_other | 6.60% | 71.56% | 75.79% |
| jsd_upstream | final | test_clean | 3.51% | 75.80% | 79.36% |
| jsd_upstream | final | test_other | 6.27% | 71.49% | 75.74% |
| jsd_upstream | final | buckeye_dev | 26.90% | 58.23% | 63.40% |
| jsd_upstream | final | buckeye_test | 30.16% | 53.86% | 59.07% |
| jsd_upstream | best_dev | dev_clean | 3.23% | 77.70% | 80.95% |
| jsd_upstream | best_dev | dev_other | 6.16% | 73.50% | 77.43% |
| jsd_upstream | best_dev | test_clean | 3.35% | 76.50% | 79.94% |
| jsd_upstream | best_dev | test_other | 5.88% | 73.33% | 77.28% |
| jsd_upstream | best_dev | buckeye_dev | 26.63% | 61.15% | 65.76% |
| jsd_upstream | best_dev | buckeye_test | 29.97% | 57.33% | 61.95% |
| jsd_upstream | average_last5 | dev_clean | 3.25% | 77.44% | 80.74% |
| jsd_upstream | average_last5 | dev_other | 6.32% | 72.60% | 76.67% |
| jsd_upstream | average_last5 | test_clean | 3.35% | 76.25% | 79.74% |
| jsd_upstream | average_last5 | test_other | 6.01% | 72.50% | 76.58% |
| jsd_upstream | average_last5 | buckeye_dev | 26.62% | 59.87% | 64.75% |
| jsd_upstream | average_last5 | buckeye_test | 29.94% | 55.66% | 60.56% |
| ce | libri100_epoch40 | dev_clean | 3.11% | 74.04% | 77.88% |
| ce | libri100_epoch40 | dev_other | 6.52% | 70.09% | 74.56% |
| ce | libri100_epoch40 | test_clean | 3.16% | 73.01% | 77.01% |
| ce | libri100_epoch40 | test_other | 5.99% | 69.92% | 74.41% |
| ce | libri100_epoch40 | buckeye_dev | 27.34% | 57.51% | 62.88% |
| ce | libri100_epoch40 | buckeye_test | 31.47% | 52.57% | 58.20% |
| ce | final | dev_clean | 4.21% | 72.28% | 76.40% |
| ce | final | dev_other | 7.93% | 67.50% | 72.38% |
| ce | final | test_clean | 4.37% | 71.12% | 75.42% |
| ce | final | test_other | 7.33% | 67.45% | 72.35% |
| ce | final | buckeye_dev | 30.78% | 53.25% | 59.78% |
| ce | final | buckeye_test | 32.80% | 49.99% | 56.44% |
| ce | best_dev | dev_clean | 3.11% | 74.04% | 77.88% |
| ce | best_dev | dev_other | 6.52% | 70.09% | 74.56% |
| ce | best_dev | test_clean | 3.16% | 73.01% | 77.01% |
| ce | best_dev | test_other | 5.99% | 69.92% | 74.41% |
| ce | best_dev | buckeye_dev | 27.34% | 57.51% | 62.88% |
| ce | best_dev | buckeye_test | 31.47% | 52.57% | 58.20% |
| ce | average_last5 | dev_clean | 3.68% | 72.95% | 76.96% |
| ce | average_last5 | dev_other | 7.18% | 68.41% | 73.15% |
| ce | average_last5 | test_clean | 3.87% | 71.85% | 76.04% |
| ce | average_last5 | test_other | 6.81% | 68.28% | 73.04% |
| ce | average_last5 | buckeye_dev | 28.81% | 54.56% | 60.72% |
| ce | average_last5 | buckeye_test | 31.74% | 51.07% | 57.26% |
| jsd_standard | libri100_epoch40 | dev_clean | 3.15% | 76.75% | 80.16% |
| jsd_standard | libri100_epoch40 | dev_other | 6.57% | 72.65% | 76.71% |
| jsd_standard | libri100_epoch40 | test_clean | 3.26% | 75.58% | 79.17% |
| jsd_standard | libri100_epoch40 | test_other | 6.05% | 72.61% | 76.67% |
| jsd_standard | libri100_epoch40 | buckeye_dev | 26.50% | 59.05% | 64.11% |
| jsd_standard | libri100_epoch40 | buckeye_test | 29.93% | 54.78% | 60.02% |
| jsd_standard | final | dev_clean | 3.19% | 75.31% | 78.95% |
| jsd_standard | final | dev_other | 6.65% | 69.46% | 74.03% |
| jsd_standard | final | test_clean | 3.35% | 74.14% | 77.96% |
| jsd_standard | final | test_other | 6.07% | 69.62% | 74.17% |
| jsd_standard | final | buckeye_dev | 27.28% | 55.23% | 60.77% |
| jsd_standard | final | buckeye_test | 30.48% | 51.09% | 56.73% |
| jsd_standard | best_dev | dev_clean | 3.12% | 76.23% | 79.72% |
| jsd_standard | best_dev | dev_other | 6.43% | 70.80% | 75.16% |
| jsd_standard | best_dev | test_clean | 3.27% | 75.04% | 78.72% |
| jsd_standard | best_dev | test_other | 6.01% | 70.83% | 75.18% |
| jsd_standard | best_dev | buckeye_dev | 26.91% | 56.46% | 61.89% |
| jsd_standard | best_dev | buckeye_test | 30.36% | 52.28% | 57.78% |
| jsd_standard | average_last5 | dev_clean | 3.12% | 76.00% | 79.52% |
| jsd_standard | average_last5 | dev_other | 6.47% | 70.37% | 74.79% |
| jsd_standard | average_last5 | test_clean | 3.28% | 74.81% | 78.53% |
| jsd_standard | average_last5 | test_other | 5.97% | 70.51% | 74.91% |
| jsd_standard | average_last5 | buckeye_dev | 27.00% | 56.10% | 61.56% |
| jsd_standard | average_last5 | buckeye_test | 30.37% | 51.87% | 57.40% |

| Objective | Variant | Split | K2 FA F1@20ms ↑ | K2 FA R-value ↑ | 共同子集边界MAE ↓ |
|---|---|---|---:|---:|---:|
| jsd_upstream | libri100_epoch40 | dev_clean | 63.87% | 69.41% | 42.43 ms |
| jsd_upstream | libri100_epoch40 | dev_other | 60.29% | 66.48% | 61.99 ms |
| jsd_upstream | libri100_epoch40 | test_clean | 62.97% | 68.67% | 49.89 ms |
| jsd_upstream | libri100_epoch40 | test_other | 60.13% | 66.35% | 66.31 ms |
| jsd_upstream | libri100_epoch40 | buckeye_dev | 54.41% | 61.31% | 30.06 ms |
| jsd_upstream | libri100_epoch40 | buckeye_test | 49.65% | 57.32% | 35.92 ms |
| jsd_upstream | final | dev_clean | 63.90% | 69.43% | 31.74 ms |
| jsd_upstream | final | dev_other | 59.30% | 65.67% | 52.46 ms |
| jsd_upstream | final | test_clean | 62.99% | 68.69% | 41.57 ms |
| jsd_upstream | final | test_other | 59.27% | 65.65% | 55.07 ms |
| jsd_upstream | final | buckeye_dev | 52.38% | 59.59% | 33.02 ms |
| jsd_upstream | final | buckeye_test | 47.83% | 55.79% | 36.87 ms |
| jsd_upstream | best_dev | dev_clean | 64.52% | 69.94% | 35.14 ms |
| jsd_upstream | best_dev | dev_other | 60.94% | 67.01% | 63.46 ms |
| jsd_upstream | best_dev | test_clean | 63.64% | 69.22% | 45.12 ms |
| jsd_upstream | best_dev | test_other | 60.70% | 66.82% | 60.39 ms |
| jsd_upstream | best_dev | buckeye_dev | 54.99% | 61.79% | 29.63 ms |
| jsd_upstream | best_dev | buckeye_test | 50.30% | 57.87% | 34.38 ms |
| jsd_upstream | average_last5 | dev_clean | 64.27% | 69.74% | 32.75 ms |
| jsd_upstream | average_last5 | dev_other | 60.11% | 66.33% | 58.60 ms |
| jsd_upstream | average_last5 | test_clean | 63.36% | 68.99% | 42.04 ms |
| jsd_upstream | average_last5 | test_other | 60.04% | 66.28% | 57.21 ms |
| jsd_upstream | average_last5 | buckeye_dev | 53.85% | 60.83% | 31.39 ms |
| jsd_upstream | average_last5 | buckeye_test | 49.16% | 56.90% | 35.34 ms |
| ce | libri100_epoch40 | dev_clean | 61.74% | 67.65% | 34.68 ms |
| ce | libri100_epoch40 | dev_other | 58.22% | 64.78% | 54.39 ms |
| ce | libri100_epoch40 | test_clean | 60.95% | 67.01% | 36.95 ms |
| ce | libri100_epoch40 | test_other | 57.95% | 64.56% | 47.98 ms |
| ce | libri100_epoch40 | buckeye_dev | 52.57% | 59.75% | 32.88 ms |
| ce | libri100_epoch40 | buckeye_test | 47.96% | 55.89% | 37.14 ms |
| ce | final | dev_clean | 60.71% | 66.81% | 31.77 ms |
| ce | final | dev_other | 56.72% | 63.56% | 45.60 ms |
| ce | final | test_clean | 59.83% | 66.09% | 34.36 ms |
| ce | final | test_other | 56.53% | 63.40% | 42.01 ms |
| ce | final | buckeye_dev | 49.18% | 56.89% | 40.51 ms |
| ce | final | buckeye_test | 45.58% | 53.89% | 43.07 ms |
| ce | best_dev | dev_clean | 61.74% | 67.65% | 34.68 ms |
| ce | best_dev | dev_other | 58.22% | 64.78% | 54.39 ms |
| ce | best_dev | test_clean | 60.95% | 67.01% | 36.95 ms |
| ce | best_dev | test_other | 57.95% | 64.56% | 47.98 ms |
| ce | best_dev | buckeye_dev | 52.57% | 59.75% | 32.88 ms |
| ce | best_dev | buckeye_test | 47.96% | 55.89% | 37.14 ms |
| ce | average_last5 | dev_clean | 61.31% | 67.30% | 30.10 ms |
| ce | average_last5 | dev_other | 57.55% | 64.24% | 42.61 ms |
| ce | average_last5 | test_clean | 60.55% | 66.68% | 32.89 ms |
| ce | average_last5 | test_other | 57.29% | 64.03% | 39.63 ms |
| ce | average_last5 | buckeye_dev | 50.56% | 58.05% | 36.51 ms |
| ce | average_last5 | buckeye_test | 46.55% | 54.71% | 40.09 ms |
| jsd_standard | libri100_epoch40 | dev_clean | 62.92% | 68.63% | 69.78 ms |
| jsd_standard | libri100_epoch40 | dev_other | 59.06% | 65.47% | 82.33 ms |
| jsd_standard | libri100_epoch40 | test_clean | 61.89% | 67.78% | 84.26 ms |
| jsd_standard | libri100_epoch40 | test_other | 58.94% | 65.38% | 88.35 ms |
| jsd_standard | libri100_epoch40 | buckeye_dev | 54.31% | 61.22% | 30.01 ms |
| jsd_standard | libri100_epoch40 | buckeye_test | 49.61% | 57.28% | 35.44 ms |
| jsd_standard | final | dev_clean | 61.75% | 67.66% | 75.48 ms |
| jsd_standard | final | dev_other | 56.94% | 63.74% | 78.71 ms |
| jsd_standard | final | test_clean | 60.85% | 66.93% | 78.33 ms |
| jsd_standard | final | test_other | 57.10% | 63.87% | 78.76 ms |
| jsd_standard | final | buckeye_dev | 50.77% | 58.23% | 35.91 ms |
| jsd_standard | final | buckeye_test | 46.64% | 54.78% | 38.90 ms |
| jsd_standard | best_dev | dev_clean | 63.02% | 68.71% | 64.83 ms |
| jsd_standard | best_dev | dev_other | 58.25% | 64.81% | 74.00 ms |
| jsd_standard | best_dev | test_clean | 61.93% | 67.82% | 77.05 ms |
| jsd_standard | best_dev | test_other | 58.29% | 64.85% | 76.68 ms |
| jsd_standard | best_dev | buckeye_dev | 52.08% | 59.34% | 32.93 ms |
| jsd_standard | best_dev | buckeye_test | 47.91% | 55.85% | 38.65 ms |
| jsd_standard | average_last5 | dev_clean | 62.61% | 68.37% | 70.45 ms |
| jsd_standard | average_last5 | dev_other | 57.81% | 64.45% | 76.87 ms |
| jsd_standard | average_last5 | test_clean | 61.59% | 67.54% | 79.94 ms |
| jsd_standard | average_last5 | test_other | 57.86% | 64.49% | 76.12 ms |
| jsd_standard | average_last5 | buckeye_dev | 51.63% | 58.95% | 34.13 ms |
| jsd_standard | average_last5 | buckeye_test | 47.53% | 55.53% | 37.43 ms |

开发集选择：{"jsd_upstream": {"variant": "best_dev", "criterion": "reference-phone-weighted FP32 development-clean/other PER only", "development_PER_percent": {"final": 4.911045268643559, "best_dev": 4.631199803730734, "average_last5": 4.714776002307781}, "test_used_for_selection": false}, "ce": {"variant": "best_dev", "criterion": "reference-phone-weighted FP32 development-clean/other PER only", "development_PER_percent": {"final": 5.986482223881764, "best_dev": 4.737152855475185, "average_last5": 5.3575039294293365}, "test_used_for_selection": false}, "jsd_standard": {"variant": "best_dev", "criterion": "reference-phone-weighted FP32 development-clean/other PER only", "development_PER_percent": {"final": 4.842297105297922, "best_dev": 4.700756768998083, "average_last5": 4.723403222806057}, "test_used_for_selection": false}}
