# CIF Buckeye 三组训练结果

核验时间：2026-10-17T06:30:06.106571+09:00

三组均保留 Quantity，权重 1；训练 40 epochs。CTC / OTTC 权重 0.5；OTTC 使用原始 upstream jsd。各组独立训练，seed 42。最终结果取 dev text-free R-value@20ms 最优 checkpoint。

边界评测：无需文本或参考 phone 数量的 CIF fires，fractional timing，容差 20 ms；排除句首句尾、保留重复 phone；冻结 IF-MDD directional greedy scorer。Buckeye 为人工 phone 边界。

| 损失 | 最佳 epoch | Dev PER | Dev P | Dev R | Dev F1 | Dev R-value | Test PER | Test P | Test R | Test F1 | Test R-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CE + Quantity + OTTC | 14.00 | 16.61 | 56.26 | 57.76 | 57.00 | 62.96 | 20.79 | 53.06 | 54.32 | 53.68 | 60.14 |
| CE + Quantity | 18.00 | 15.97 | 56.01 | 56.42 | 56.22 | 62.54 | 20.22 | 51.28 | 50.74 | 51.01 | 58.32 |
| CE + Quantity + CTC | 10.00 | 18.01 | 57.19 | 59.24 | 58.20 | 63.86 | 20.63 | 53.51 | 54.91 | 54.20 | 60.55 |

百分比单位。三组 final dev 1524 / test 1840 条预测均完整且无重复，评测 ID 集合一致；六个额外 benchmark 的预测覆盖率也全部通过核验。所有结果为单 seed。

OTTC 相对 CE+Quantity 的 test F1 提高 2.67 个百分点、R-value 提高 1.82 个百分点；CTC 比 OTTC 的 F1 高 0.52、R-value 高 0.42 个百分点。CE+Quantity 的 PER 最低。

Libri100 三组尚未完成；源边界为自动对齐参考，不能等同于人工边界真值。跨训练语料对比需保留独立数据域与 scorer 协议。

## W&B

- CE + Quantity + OTTC: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/sqp2q4q9
- CE + Quantity: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/5za3yqhg
- CE + Quantity + CTC: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/wwhhvujw

详细数值、协议、覆盖率及当前 Libri100 进度：`status/track_20261017_063010_verified.json`。
