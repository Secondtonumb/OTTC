# CIF 三语料 40 epoch 最终结果核验

核验时间：2026-10-17T14:44:09.834048+09:00。TIMIT、Buckeye、Libri100 各三组，共九组均完成 40 epoch。逐组核对 TRAIN_COMPLETE、训练步数、最终 dev/test 指标及预测数量和 ID 唯一性。

所有组保留 CE + Quantity（Quantity 权重 1）；辅助 OTTC / CTC 权重 0.5，OTTC 为原始 upstream jsd。各组按 dev text-free R-value@20ms 选择单个 checkpoint。下表单位为百分比，PER 为去静音版本。

|语料|辅助损失|最佳 epoch|split|PER ↓|P|Recall|F1|R-value ↑|
|---|---|---:|---|---:|---:|---:|---:|---:|
|timit|ce|10|dev|13.961|57.245|58.604|57.917|63.793|
|timit|ce|10|test|14.557|59.117|60.787|59.940|65.480|
|timit|ctc|24|dev|8.770|58.853|57.824|58.334|64.601|
|timit|ctc|24|test|10.139|58.661|57.737|58.195|64.468|
|timit|ottc|11|dev|13.927|57.137|56.580|56.857|63.278|
|timit|ottc|11|test|15.861|57.513|57.077|57.294|63.628|
|buckeye|ottc|14|dev|16.612|56.261|57.762|57.002|62.962|
|buckeye|ottc|14|test|20.793|53.057|54.317|53.680|60.137|
|buckeye|ce|18|dev|15.970|56.010|56.425|56.217|62.541|
|buckeye|ce|18|test|20.225|51.276|50.739|51.006|58.318|
|buckeye|ctc|10|dev|18.008|57.187|59.242|58.196|63.861|
|buckeye|ctc|10|test|20.632|53.513|54.912|54.204|60.553|
|libri100|ottc|3|dev|6.278|55.275|54.866|55.070|61.734|
|libri100|ottc|3|test|6.122|54.654|54.290|54.471|61.217|
|libri100|ce|3|dev|7.117|55.304|56.108|55.703|62.010|
|libri100|ce|3|test|7.191|54.451|55.297|54.871|61.281|
|libri100|ctc|2|dev|7.559|53.894|53.728|53.811|60.613|
|libri100|ctc|2|test|7.637|53.021|52.908|52.964|59.879|

无文本分段协议：fractional duration/T 时钟，20ms 容差，排除句首句尾、保留重复 phone，冻结 IF-MDD directional greedy 与 R-value 公式。Libri 边界参考是自动对齐且 aligner 身份未验证；Buckeye 是人工 phone 边界。各语料分开解释。

本轮单 seed：Buckeye 的 OTTC 辅助相对 CE+Quantity 改善分段，但稍低于 CTC；Libri100 的 OTTC 辅助识别更好，分段与 CE+Quantity 接近；TIMIT 的 OTTC 辅助未见收益。

TIMIT 与历史原生 OTTC 的统一口径比较另见 reproduction/cif_comparison_20261016/REPORT_ZH.md，不能将此处原生时钟数字与历史固定帧时钟数字混报。

当前九组仅保存 best/latest，无逐 epoch 权重，无法做 last-5/last-10 平均。没有重启训练。旧 Libri 中期评测报告保留其原有语义；本报告使用训练完成后生成的 final_dev/final_test。

## W&B

- timit / ce: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/0zv0ppad
- timit / ctc: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/vxs9qj1o
- timit / ottc: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/74go8z0b
- buckeye / ottc: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/sqp2q4q9
- buckeye / ce: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/5za3yqhg
- buckeye / ctc: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/wwhhvujw
- libri100 / ottc: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/b82bcq1m
- libri100 / ce: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/u61cpmzd
- libri100 / ctc: https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/runs/giceiaut

完整核验记录：reproduction/cif_quantity_20261016/status/final_verified_20261017.json。
