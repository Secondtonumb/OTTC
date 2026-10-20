# 三个 checkpoint 的比较

更新：2026-10-10T07:04:38.829745+09:00。

比较最后三个逐个测试、最后三个权重平均，以及 dev PER 最好的三个权重平均。最后三个指已保存 checkpoint，可能相隔一两个更新，并非三个 epoch。权重均匀平均；不平均 optimizer/RNG。四头按原始 OTTC ARPAbet dev PER 选，MFA 按原生 phone dev PER 选；test 不参与选择。

TIMIT 只剩最后三个权重，因此“最好三个”只能在这三个中选，与最后三个平均相同，不能称为全程最好三个。Libri100 现存及以后权重已增加独立保存；中途快照与最终结果单独标记。MFA 对全部现存 checkpoint 补 dev 后选三个。

| 数据 / 目标 | 状态 | 候选数 | Job |
|---|---|---:|---|
| libri100_ce | ✅ 已完成 | 4 | 3512789.opbs |
| libri100_ctc | ✅ 已完成 | 3 | 3512788.opbs |
| libri100_jsd_upstream | ✅ 已完成 | 3 | 3512790.opbs |
| mfa_ctc | ✅ 已完成 | 17 | 3512785.opbs |
| mfa_ottc | ✅ 已完成 | 15 | 3512786.opbs |
| mfa_spottc | ✅ 已完成 | 16 | 3512787.opbs |
| timit_ce | ✅ 已完成 | 3 | 3512801.opbs |
| timit_ctc | ✅ 已完成 | 3 | 3512800.opbs |
| timit_jsd_upstream | ✅ 已完成 | 3 | 3512802.opbs |

## libri100_ce（中途 checkpoint）

最后三个：[15164, 15610, 16056]；最好三个：[15610, 16056, 15164]；dev 最好单个：15610。

[W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/vkl0wfbj)

| 策略 / step | head | Test PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |
|---|---|---:|---:|---:|---:|---:|
| step 15164 | ottc_arpabet | 4.50 | 71.40 | 75.65 | 57.83 | 64.30 |
| step 15164 | ifmdd_arpabet | 5.04 | 71.16 | 75.45 | 57.70 | 64.19 |
| step 15164 | mfa_ipa | 4.30 | — | — | — | — |
| step 15164 | zipa_ipa | 3.56 | — | — | — | — |
| step 15610 | ottc_arpabet | 4.44 | 71.46 | 75.70 | 58.06 | 64.49 |
| step 15610 | ifmdd_arpabet | 4.97 | 71.23 | 75.52 | 57.95 | 64.40 |
| step 15610 | mfa_ipa | 4.20 | — | — | — | — |
| step 15610 | zipa_ipa | 3.53 | — | — | — | — |
| step 16056 | ottc_arpabet | 4.46 | 71.38 | 75.63 | 57.85 | 64.32 |
| step 16056 | ifmdd_arpabet | 4.99 | 71.13 | 75.43 | 57.71 | 64.20 |
| step 16056 | mfa_ipa | 4.19 | — | — | — | — |
| step 16056 | zipa_ipa | 3.56 | — | — | — | — |
| last3_weight_average | ottc_arpabet | 4.44 | 71.42 | 75.66 | 57.92 | 64.37 |
| last3_weight_average | ifmdd_arpabet | 4.98 | 71.18 | 75.47 | 57.80 | 64.27 |
| last3_weight_average | mfa_ipa | 4.21 | — | — | — | — |
| last3_weight_average | zipa_ipa | 3.55 | — | — | — | — |
| best3_weight_average | ottc_arpabet | 4.44 | 71.42 | 75.66 | 57.92 | 64.37 |
| best3_weight_average | ifmdd_arpabet | 4.98 | 71.18 | 75.47 | 57.80 | 64.27 |
| best3_weight_average | mfa_ipa | 4.21 | — | — | — | — |
| best3_weight_average | zipa_ipa | 3.55 | — | — | — | — |

逐个模型的指标均值及标准差见 comparison.json；它们与权重平均模型的重新测试分开记录。

## libri100_ctc（中途 checkpoint）

最后三个：[16502, 16948, 17394]；最好三个：[17394, 16948, 16502]；dev 最好单个：17394。

[W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/cbmvtuei)

| 策略 / step | head | Test PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |
|---|---|---:|---:|---:|---:|---:|
| step 16502 | ottc_arpabet | 3.67 | 55.31 | 33.88 | 52.84 | 60.12 |
| step 16502 | ifmdd_arpabet | 3.69 | 55.52 | 34.22 | 52.82 | 60.10 |
| step 16502 | mfa_ipa | 2.41 | — | — | — | — |
| step 16502 | zipa_ipa | 2.38 | — | — | — | — |
| step 16948 | ottc_arpabet | 3.66 | 55.28 | 33.79 | 52.83 | 60.11 |
| step 16948 | ifmdd_arpabet | 3.68 | 55.48 | 34.14 | 52.81 | 60.10 |
| step 16948 | mfa_ipa | 2.42 | — | — | — | — |
| step 16948 | zipa_ipa | 2.38 | — | — | — | — |
| step 17394 | ottc_arpabet | 3.65 | 55.25 | 33.80 | 52.81 | 60.09 |
| step 17394 | ifmdd_arpabet | 3.67 | 55.46 | 34.14 | 52.80 | 60.08 |
| step 17394 | mfa_ipa | 2.40 | — | — | — | — |
| step 17394 | zipa_ipa | 2.35 | — | — | — | — |
| last3_weight_average | ottc_arpabet | 3.66 | 55.28 | 33.82 | 52.83 | 60.11 |
| last3_weight_average | ifmdd_arpabet | 3.67 | 55.49 | 34.17 | 52.81 | 60.10 |
| last3_weight_average | mfa_ipa | 2.41 | — | — | — | — |
| last3_weight_average | zipa_ipa | 2.37 | — | — | — | — |
| best3_weight_average | ottc_arpabet | 3.66 | 55.28 | 33.82 | 52.83 | 60.11 |
| best3_weight_average | ifmdd_arpabet | 3.67 | 55.49 | 34.17 | 52.81 | 60.10 |
| best3_weight_average | mfa_ipa | 2.41 | — | — | — | — |
| best3_weight_average | zipa_ipa | 2.37 | — | — | — | — |

逐个模型的指标均值及标准差见 comparison.json；它们与权重平均模型的重新测试分开记录。

## libri100_jsd_upstream（中途 checkpoint）

最后三个：[15164, 15610, 16056]；最好三个：[16056, 15610, 15164]；dev 最好单个：16056。

[W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/f88q3c90)

| 策略 / step | head | Test PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |
|---|---|---:|---:|---:|---:|---:|
| step 15164 | ottc_arpabet | 4.81 | 73.78 | 77.66 | 62.35 | 68.08 |
| step 15164 | ifmdd_arpabet | 5.33 | 73.60 | 77.51 | 62.20 | 67.96 |
| step 15164 | mfa_ipa | 4.84 | — | — | — | — |
| step 15164 | zipa_ipa | 3.77 | — | — | — | — |
| step 15610 | ottc_arpabet | 4.68 | 73.83 | 77.70 | 62.13 | 67.90 |
| step 15610 | ifmdd_arpabet | 5.20 | 73.65 | 77.55 | 61.99 | 67.79 |
| step 15610 | mfa_ipa | 4.59 | — | — | — | — |
| step 15610 | zipa_ipa | 3.66 | — | — | — | — |
| step 16056 | ottc_arpabet | 4.63 | 73.68 | 77.57 | 62.19 | 67.95 |
| step 16056 | ifmdd_arpabet | 5.15 | 73.50 | 77.42 | 62.08 | 67.86 |
| step 16056 | mfa_ipa | 4.49 | — | — | — | — |
| step 16056 | zipa_ipa | 3.60 | — | — | — | — |
| last3_weight_average | ottc_arpabet | 4.70 | 73.77 | 77.65 | 62.22 | 67.97 |
| last3_weight_average | ifmdd_arpabet | 5.23 | 73.60 | 77.51 | 62.09 | 67.86 |
| last3_weight_average | mfa_ipa | 4.65 | — | — | — | — |
| last3_weight_average | zipa_ipa | 3.66 | — | — | — | — |
| best3_weight_average | ottc_arpabet | 4.70 | 73.77 | 77.65 | 62.22 | 67.97 |
| best3_weight_average | ifmdd_arpabet | 5.23 | 73.60 | 77.51 | 62.09 | 67.86 |
| best3_weight_average | mfa_ipa | 4.65 | — | — | — | — |
| best3_weight_average | zipa_ipa | 3.66 | — | — | — | — |

逐个模型的指标均值及标准差见 comparison.json；它们与权重平均模型的重新测试分开记录。

## mfa_ctc（中途 checkpoint）

最后三个：[24514, 26265, 26266]；最好三个：[21003, 21012, 14008]；dev 最好单个：21003。

[W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/dyhwdjsk)

| 策略 / step | head | Test PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |
|---|---|---:|---:|---:|---:|---:|
| step 24514 | MFA phones | 45.12 | 40.05 | 22.21 | 41.98 | 50.50 |
| step 26265 | MFA phones | 44.04 | 40.51 | 19.72 | 48.02 | 55.66 |
| step 26266 | MFA phones | 44.08 | 40.48 | 20.11 | 47.99 | 55.63 |
| step 21003 | MFA phones | 41.74 | 41.70 | 22.27 | 45.02 | 53.08 |
| step 21012 | MFA phones | 42.20 | 41.78 | 22.05 | 44.77 | 52.86 |
| step 14008 | MFA phones | 41.84 | 42.91 | 23.36 | 47.62 | 55.30 |
| last3_weight_average | MFA phones | 42.18 | 43.96 | 23.02 | 51.13 | 58.31 |
| best3_weight_average | MFA phones | 40.61 | 43.11 | 21.94 | 46.36 | 54.23 |

逐个模型的指标均值及标准差见 comparison.json；它们与权重平均模型的重新测试分开记录。

## mfa_ottc（中途 checkpoint）

最后三个：[21012, 22763, 24287]；最好三个：[19261, 12257, 5353]；dev 最好单个：19261。

[W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/6fgwh1aq)

| 策略 / step | head | Test PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |
|---|---|---:|---:|---:|---:|---:|
| step 21012 | MFA phones | 78.16 | 44.32 | 42.57 | 39.90 | 49.72 |
| step 22763 | MFA phones | 61.14 | 46.39 | 51.21 | 39.52 | 49.44 |
| step 24287 | MFA phones | 64.46 | 44.07 | 48.52 | 37.83 | 48.06 |
| step 19261 | MFA phones | 56.28 | 46.98 | 52.69 | 40.27 | 50.06 |
| step 12257 | MFA phones | 59.39 | 43.61 | 49.04 | 38.18 | 48.35 |
| step 5353 | MFA phones | 60.44 | 41.54 | 47.09 | 35.21 | 45.92 |
| last3_weight_average | MFA phones | 49.54 | 47.63 | 54.51 | 40.96 | 50.61 |
| best3_weight_average | MFA phones | 83.05 | 39.68 | 36.55 | 38.37 | 48.53 |

逐个模型的指标均值及标准差见 comparison.json；它们与权重平均模型的重新测试分开记录。

## mfa_spottc（中途 checkpoint）

评测与 W&B 上传已完成；原 PBS 因 Markdown 汇总字段冲突 exit1。已读取冻结 selection 文件恢复汇总，未重跑 GPU 或修改评分。

最后三个：[20664, 22386, 22643]；最好三个：[22386, 20664, 20662]；dev 最好单个：22386。

[W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/2a9u1d3m)

| 策略 / step | head | Test PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |
|---|---|---:|---:|---:|---:|---:|
| step 20664 | MFA phones | 45.16 | 41.00 | 8.25 | 39.70 | 48.80 |
| step 22386 | MFA phones | 45.09 | 40.84 | 11.00 | 39.71 | 48.79 |
| step 22643 | MFA phones | 45.86 | 41.35 | 8.33 | 40.37 | 49.36 |
| step 20662 | MFA phones | 45.16 | 41.13 | 8.66 | 39.86 | 48.93 |
| last3_weight_average | MFA phones | 44.52 | 41.37 | 7.44 | 39.97 | 49.02 |
| best3_weight_average | MFA phones | 44.19 | 41.49 | 7.49 | 39.42 | 48.56 |

逐个模型的指标均值及标准差见 comparison.json；它们与权重平均模型的重新测试分开记录。

## timit_ce（训练完成）

最后三个：[2774, 2847, 2920]；最好三个：[2847, 2920, 2774]；dev 最好单个：2847。

[W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/kbka0y87)

| 策略 / step | head | Test PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |
|---|---|---:|---:|---:|---:|---:|
| step 2774 | ottc_arpabet | 9.89 | 63.64 | 68.62 | 60.84 | 66.58 |
| step 2774 | ifmdd_arpabet | 9.90 | 63.20 | 68.49 | 60.43 | 66.37 |
| step 2774 | mfa_ipa | 11.36 | — | — | — | — |
| step 2774 | zipa_ipa | 13.01 | — | — | — | — |
| step 2847 | ottc_arpabet | 9.91 | 63.82 | 68.78 | 60.91 | 66.64 |
| step 2847 | ifmdd_arpabet | 9.77 | 63.42 | 68.68 | 60.60 | 66.52 |
| step 2847 | mfa_ipa | 11.34 | — | — | — | — |
| step 2847 | zipa_ipa | 12.84 | — | — | — | — |
| step 2920 | ottc_arpabet | 9.94 | 63.62 | 68.60 | 60.77 | 66.52 |
| step 2920 | ifmdd_arpabet | 9.84 | 63.25 | 68.51 | 60.37 | 66.33 |
| step 2920 | mfa_ipa | 11.29 | — | — | — | — |
| step 2920 | zipa_ipa | 12.89 | — | — | — | — |
| last3_weight_average | ottc_arpabet | 9.91 | 63.75 | 68.71 | 60.95 | 66.68 |
| last3_weight_average | ifmdd_arpabet | 9.83 | 63.28 | 68.55 | 60.49 | 66.42 |
| last3_weight_average | mfa_ipa | 11.38 | — | — | — | — |
| last3_weight_average | zipa_ipa | 13.03 | — | — | — | — |
| best3_weight_average | ottc_arpabet | 9.91 | 63.75 | 68.71 | 60.95 | 66.68 |
| best3_weight_average | ifmdd_arpabet | 9.83 | 63.28 | 68.55 | 60.49 | 66.42 |
| best3_weight_average | mfa_ipa | 11.38 | — | — | — | — |
| best3_weight_average | zipa_ipa | 13.03 | — | — | — | — |

逐个模型的指标均值及标准差见 comparison.json；它们与权重平均模型的重新测试分开记录。

## timit_ctc（训练完成）

最后三个：[2774, 2847, 2920]；最好三个：[2920, 2847, 2774]；dev 最好单个：2920。

[W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/uw3wdqy2)

| 策略 / step | head | Test PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |
|---|---|---:|---:|---:|---:|---:|
| step 2774 | ottc_arpabet | 10.28 | 52.85 | 19.78 | 51.79 | 58.87 |
| step 2774 | ifmdd_arpabet | 9.08 | 52.94 | 20.38 | 51.58 | 58.89 |
| step 2774 | mfa_ipa | 19.53 | — | — | — | — |
| step 2774 | zipa_ipa | 9.47 | — | — | — | — |
| step 2847 | ottc_arpabet | 10.28 | 52.88 | 19.94 | 51.92 | 58.97 |
| step 2847 | ifmdd_arpabet | 9.02 | 53.01 | 20.61 | 51.69 | 58.99 |
| step 2847 | mfa_ipa | 19.49 | — | — | — | — |
| step 2847 | zipa_ipa | 9.43 | — | — | — | — |
| step 2920 | ottc_arpabet | 10.29 | 52.93 | 19.99 | 51.90 | 58.96 |
| step 2920 | ifmdd_arpabet | 9.05 | 53.05 | 20.64 | 51.71 | 59.00 |
| step 2920 | mfa_ipa | 19.55 | — | — | — | — |
| step 2920 | zipa_ipa | 9.47 | — | — | — | — |
| last3_weight_average | ottc_arpabet | 10.32 | 52.90 | 19.95 | 51.90 | 58.96 |
| last3_weight_average | ifmdd_arpabet | 9.04 | 53.01 | 20.57 | 51.65 | 58.95 |
| last3_weight_average | mfa_ipa | 19.62 | — | — | — | — |
| last3_weight_average | zipa_ipa | 9.43 | — | — | — | — |
| best3_weight_average | ottc_arpabet | 10.32 | 52.90 | 19.95 | 51.90 | 58.96 |
| best3_weight_average | ifmdd_arpabet | 9.04 | 53.01 | 20.57 | 51.65 | 58.95 |
| best3_weight_average | mfa_ipa | 19.62 | — | — | — | — |
| best3_weight_average | zipa_ipa | 9.43 | — | — | — | — |

逐个模型的指标均值及标准差见 comparison.json；它们与权重平均模型的重新测试分开记录。

## timit_jsd_upstream（训练完成）

最后三个：[2774, 2847, 2920]；最好三个：[2774, 2920, 2847]；dev 最好单个：2774。

[W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/n2xcz9e9)

| 策略 / step | head | Test PER / token error % ↓ | Text-free F1@20ms % ↑ | Text-free R-value % ↑ | K2 FA F1@20ms % ↑ | K2 FA R-value % ↑ |
|---|---|---:|---:|---:|---:|---:|
| step 2774 | ottc_arpabet | 9.69 | 67.72 | 72.15 | 65.09 | 70.21 |
| step 2774 | ifmdd_arpabet | 9.51 | 67.20 | 71.94 | 64.44 | 69.76 |
| step 2774 | mfa_ipa | 11.23 | — | — | — | — |
| step 2774 | zipa_ipa | 19.93 | — | — | — | — |
| step 2847 | ottc_arpabet | 9.63 | 67.61 | 72.06 | 64.76 | 69.93 |
| step 2847 | ifmdd_arpabet | 9.54 | 67.14 | 71.88 | 64.14 | 69.51 |
| step 2847 | mfa_ipa | 11.27 | — | — | — | — |
| step 2847 | zipa_ipa | 19.76 | — | — | — | — |
| step 2920 | ottc_arpabet | 9.68 | 67.65 | 72.10 | 64.76 | 69.93 |
| step 2920 | ifmdd_arpabet | 9.51 | 67.07 | 71.82 | 64.13 | 69.50 |
| step 2920 | mfa_ipa | 11.29 | — | — | — | — |
| step 2920 | zipa_ipa | 19.82 | — | — | — | — |
| last3_weight_average | ottc_arpabet | 9.72 | 67.64 | 72.08 | 64.81 | 69.97 |
| last3_weight_average | ifmdd_arpabet | 9.58 | 67.01 | 71.77 | 64.13 | 69.50 |
| last3_weight_average | mfa_ipa | 11.23 | — | — | — | — |
| last3_weight_average | zipa_ipa | 19.90 | — | — | — | — |
| best3_weight_average | ottc_arpabet | 9.72 | 67.64 | 72.08 | 64.81 | 69.97 |
| best3_weight_average | ifmdd_arpabet | 9.58 | 67.01 | 71.77 | 64.13 | 69.50 |
| best3_weight_average | mfa_ipa | 11.23 | — | — | — | — |
| best3_weight_average | zipa_ipa | 19.90 | — | — | — | — |

逐个模型的指标均值及标准差见 comparison.json；它们与权重平均模型的重新测试分开记录。
