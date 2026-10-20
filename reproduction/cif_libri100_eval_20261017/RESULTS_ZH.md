# Libri100 CIF 中期 checkpoint 测试结果

核验时间：2026-10-17T08:12:37.920375+09:00

这是训练进行中的独立评测：三组均保留 Quantity（权重 1），CTC / OTTC 辅助权重 0.5；OTTC 使用原始 upstream jsd。各组选取测试开始前 dev 分段 R-value@20ms 最优 checkpoint，测试集未参与选点。现有训练继续至 40 epochs。

分段无需参考文本或 phone 数量；fractional duration/T 时钟，20ms 容差，排除句首句尾，保留重复 phone，冻结 IF-MDD directional greedy scorer。Precision、Recall、F1、R-value、PER 的单位均为百分比。

## Libri 合并 test（5558 条）

| 损失 | 选中 epoch | PER raw | PER nosil | Precision | Recall | F1 | R-value |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE + Quantity + OTTC | 3 | 6.465 | 6.122 | 54.654 | 54.290 | 54.471 | 61.217 |
| CE + Quantity | 3 | 7.371 | 7.191 | 54.451 | 55.297 | 54.871 | 61.281 |
| CE + Quantity + CTC | 2 | 7.720 | 7.637 | 53.021 | 52.908 | 52.964 | 59.879 |

## Libri test-clean（2620 条）

| 损失 | 选中 epoch | PER raw | PER nosil | Precision | Recall | F1 | R-value |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE + Quantity + OTTC | 3 | 4.914 | 4.579 | 54.705 | 54.456 | 54.580 | 61.286 |
| CE + Quantity | 3 | 6.137 | 5.948 | 54.294 | 55.157 | 54.722 | 61.148 |
| CE + Quantity + CTC | 2 | 5.928 | 5.782 | 52.760 | 52.802 | 52.781 | 59.686 |

## Libri test-other（2938 条）

| 损失 | 选中 epoch | PER raw | PER nosil | Precision | Recall | F1 | R-value |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE + Quantity + OTTC | 3 | 8.077 | 7.733 | 54.600 | 54.117 | 54.357 | 61.145 |
| CE + Quantity | 3 | 8.653 | 8.488 | 54.615 | 55.444 | 55.026 | 61.420 |
| CE + Quantity + CTC | 2 | 9.581 | 9.573 | 53.294 | 53.019 | 53.156 | 60.080 |

## Buckeye 人工边界 test（1840 条；Libri 训练模型跨语料评测）

| 损失 | 选中 epoch | PER raw | PER nosil | Precision | Recall | F1 | R-value |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE + Quantity + OTTC | 3 | 37.352 | 30.652 | 49.239 | 53.640 | 51.345 | 56.842 |
| CE + Quantity | 3 | 39.045 | 32.470 | 49.234 | 54.864 | 51.897 | 56.718 |
| CE + Quantity + CTC | 2 | 39.082 | 32.534 | 49.470 | 52.581 | 50.978 | 57.094 |

三组完整 Libri/Buckeye 预测数量、ID 唯一性、跨组参考 phone 序列及边界一致性均通过核验。10/30ms 和 right-cell grid 指标见各组 JSON。Libri 边界参考为自动对齐、aligner 身份未验证；Buckeye 为人工 phone 边界，两域分别报告。

当前实验没有保存逐 epoch 权重，不能据此提供 last-5 / last-10 checkpoint 平均结果。这些是单个 best-dev checkpoint 的结果。

## W&B

- CE + Quantity + OTTC: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/bwspd14v
- CE + Quantity: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/skin4w1g
- CE + Quantity + CTC: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/6eg2a7hc

来源模型、数据、评测器及预测 SHA256，完整指标和选点信息见 `RESULTS.json`。
