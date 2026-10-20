# Libri100 单头 OTTC：CE / 原上游 JSD / standard JSD

状态：三组最终评测完成。

train-clean-100 全部 28,538句，100.59小时；官方说话人隔离。三组单头 wav2vec2-large-lv60 自监督预训练、随机phone/mass头，CNN冻结、Transformer微调；seed42、batch64、lr2e-4、AdamW weight decay0.005、warmup500、BF16/FP32、40epoch/17840step、regfrozen30epoch。固定最后epoch40评测。

复用 Buckeye 的原 OTTC 实现，只改变现有运输代价分支。原上游JSD为0.5 KL(m||p)+0.5 KL(m||q)，沿用target epsilon；standard为0.5 KL(p||m)+0.5 KL(q||m)。raw loss定义及尺度不同，不直接排名。

Libri clean/other 参考来自已有自动对齐（模型身份未核实），不能称作人工分割精度。Buckeye dev/test 为同一原始人工标注19位子集上的零样本评测，可直接比较 Buckeye-trained 三组；它不参与Libri训练或checkpoint选择。

参考speech-phone区间只映射ao→aa、zh→sh，不合并；去sil/q。训练目标复用canonical ottc_arpabet序列（含sil、合并相邻重复），缓存只含音频/标签/长度，无时间戳。

Text-free与给定音素K2 FA分别报告；FA不使用参考时间戳或sil目标。FP32推理、20ms网格；FA为duration/frame_count、blank归前一音素、无偏移。边界为唯一区间端点、去句首句尾20ms、一对一时间匹配。R-value沿用冻结IF-MDD公式并保存哈希。F1/R使用全集，失败零命中；MAE为完成组共同成功子集，全部完成后为三组交集。

| Split | OTTC代价 | Text-free PER ↓ | Text-free F1@20ms ↑ | Text-free R-value ↑ |
|---|---|---:|---:|---:|
| dev_clean | CE | 3.11% | 74.04% | 77.88% |
| dev_clean | 原上游 JSD | 3.23% | 77.57% | 80.84% |
| dev_clean | standard JSD | 3.15% | 76.75% | 80.16% |
| dev_other | CE | 6.52% | 70.09% | 74.56% |
| dev_other | 原上游 JSD | 6.88% | 73.39% | 77.33% |
| dev_other | standard JSD | 6.57% | 72.65% | 76.71% |
| test_clean | CE | 3.16% | 73.01% | 77.01% |
| test_clean | 原上游 JSD | 3.35% | 76.38% | 79.84% |
| test_clean | standard JSD | 3.26% | 75.58% | 79.17% |
| test_other | CE | 5.99% | 69.92% | 74.41% |
| test_other | 原上游 JSD | 6.22% | 73.39% | 77.33% |
| test_other | standard JSD | 6.05% | 72.61% | 76.67% |
| buckeye_dev | CE | 27.34% | 57.51% | 62.88% |
| buckeye_dev | 原上游 JSD | 26.79% | 61.03% | 65.81% |
| buckeye_dev | standard JSD | 26.50% | 59.05% | 64.11% |
| buckeye_test | CE | 31.47% | 52.57% | 58.20% |
| buckeye_test | 原上游 JSD | 31.15% | 56.75% | 61.58% |
| buckeye_test | standard JSD | 29.93% | 54.78% | 60.02% |

| Split | OTTC代价 | K2 FA F1@20ms ↑ | K2 FA R-value ↑ | 共同子集边界MAE ↓ | 对齐失败 |
|---|---|---:|---:|---:|---:|
| dev_clean | CE | 61.74% | 67.65% | 34.68 ms | 0/2703 |
| dev_clean | 原上游 JSD | 63.87% | 69.41% | 42.43 ms | 0/2703 |
| dev_clean | standard JSD | 62.92% | 68.63% | 69.78 ms | 0/2703 |
| dev_other | CE | 58.22% | 64.78% | 54.39 ms | 1/2864 |
| dev_other | 原上游 JSD | 60.29% | 66.48% | 61.99 ms | 1/2864 |
| dev_other | standard JSD | 59.06% | 65.47% | 82.33 ms | 1/2864 |
| test_clean | CE | 60.95% | 67.01% | 36.95 ms | 0/2620 |
| test_clean | 原上游 JSD | 62.97% | 68.67% | 49.89 ms | 0/2620 |
| test_clean | standard JSD | 61.89% | 67.78% | 84.26 ms | 0/2620 |
| test_other | CE | 57.95% | 64.56% | 47.98 ms | 0/2938 |
| test_other | 原上游 JSD | 60.13% | 66.35% | 66.31 ms | 0/2938 |
| test_other | standard JSD | 58.94% | 65.38% | 88.35 ms | 0/2938 |
| buckeye_dev | CE | 52.57% | 59.75% | 32.88 ms | 0/1524 |
| buckeye_dev | 原上游 JSD | 54.41% | 61.31% | 30.06 ms | 0/1524 |
| buckeye_dev | standard JSD | 54.31% | 61.22% | 30.01 ms | 0/1524 |
| buckeye_test | CE | 47.96% | 55.89% | 37.14 ms | 0/1840 |
| buckeye_test | 原上游 JSD | 49.65% | 57.32% | 35.92 ms | 0/1840 |
| buckeye_test | standard JSD | 49.61% | 57.28% | 35.44 ms | 0/1840 |

同一人工标注 Buckeye test：训练语料对照（完整集合，未做checkpoint平均）

| OTTC代价 | 训练语料 | Text-free PER ↓ | Text-free F1@20ms ↑ | Text-free R-value ↑ | K2 FA F1@20ms ↑ | K2 FA R-value ↑ |
|---|---|---:|---:|---:|---:|---:|
| CE | Buckeye | 20.19% | 52.00% | 57.47% | 49.45% | 57.15% |
| CE | Libri100 | 31.47% | 52.57% | 58.20% | 47.96% | 55.89% |
| 原上游 JSD | Buckeye | 19.64% | 54.06% | 58.97% | 49.93% | 57.55% |
| 原上游 JSD | Libri100 | 31.15% | 56.75% | 61.58% | 49.65% | 57.32% |
| standard JSD | Buckeye | 21.15% | 53.67% | 58.78% | 50.55% | 58.07% |
| standard JSD | Libri100 | 29.93% | 54.78% | 60.02% | 49.61% | 57.28% |

完整P/R/F1/R、10/20/30ms、逐句路径与来源位于evaluation目录，训练与最终评测均记录W&B；checkpoint平均尚未包含。
