# 已跑实验标记

更新：2026-10-12T03:49:38+09:00。

✅ 完成；🟡 已实际训练但尚未完成；⏳ 尚未开始正式训练。MFA 标签英语全量指 Zapformer 系列，四头里的 MFA IPA head 共享 Wav2Vec2，属于另一系列。

CE：已完成且数据/模型配置相同的实验直接复用权重，不重训；未完成的从断点续训。checkpoint 比较仅做推理、参数平均和评分，不执行训练更新。

| 实验系列 | 目标 | 前端 / 帧步长 | 训练状态 | 已有测试 | 补充评估 |
|---|---|---|---|---|---|
| 原始单头 TIMIT | CTC | Wav2Vec2 LV60 / 20ms | ✅ 已完成训练与 test | ✅ README240、core192、Buckeye、L2-ARCTIC | 历史系列未做 K2 / checkpoint 平均 |
| 原始单头 TIMIT | OTTC CE | Wav2Vec2 LV60 / 20ms | ✅ 已完成训练与 test | ✅ README240、core192、Buckeye、L2-ARCTIC | 历史系列未做 K2 / checkpoint 平均 |
| 原始单头 TIMIT | OTTC 上游 JSD | Wav2Vec2 LV60 / 20ms | ✅ 已完成训练与 test | ✅ README240、core192、Buckeye、L2-ARCTIC | 历史系列未做 K2 / checkpoint 平均 |
| 原始单头 TIMIT | OTTC 标准 JSD | Wav2Vec2 LV60 / 20ms | ✅ 已完成训练与 test | ✅ README240、core192、Buckeye、L2-ARCTIC | 历史系列未做 K2 / checkpoint 平均 |
| 原始单头 TIMIT | Soft-DTW CE | Wav2Vec2 LV60 / 20ms | ✅ 已完成训练与 test | ✅ README240、core192、Buckeye、L2-ARCTIC | 历史系列未做 K2 / checkpoint 平均 |
| timit 四头 | CTC | Wav2Vec2 LV60 / 20ms | ✅ 已完成 40 轮 | ✅ 最终 text-free test | ✅ K2 FA；✅ 三个 checkpoint 比较 |
| timit 四头 | OTTC CE | Wav2Vec2 LV60 / 20ms | ✅ 已完成 40 轮 | ✅ 最终 text-free test | ✅ K2 FA；✅ 三个 checkpoint 比较 |
| timit 四头 | OTTC 上游 JSD | Wav2Vec2 LV60 / 20ms | ✅ 已完成 40 轮 | ✅ 最终 text-free test | ✅ K2 FA；✅ 三个 checkpoint 比较 |
| libri100 四头 | CTC | Wav2Vec2 LV60 / 20ms | ✅ 已完成 40 轮 | ✅ 最终 text-free test | ✅ K2 FA；✅ 三个 checkpoint 比较 |
| libri100 四头 | OTTC CE | Wav2Vec2 LV60 / 20ms | ✅ 已完成 40 轮 | ✅ 最终 text-free test | ✅ K2 FA；✅ 三个 checkpoint 比较 |
| libri100 四头 | OTTC 上游 JSD | Wav2Vec2 LV60 / 20ms | ✅ 已完成 40 轮 | ✅ 最终 text-free test | ✅ K2 FA；✅ 三个 checkpoint 比较；✅ FA-bench clean TIMIT / processed Buckeye |
| libri960 四头 | CTC | Wav2Vec2 LV60 / 20ms | 🟡 运行中 20.20/40 | 训练完成后 test | K2 FA 待测；— |
| libri960 四头 | OTTC CE | Wav2Vec2 LV60 / 20ms | 🟡 运行中 18.93/40 | 训练完成后 test | K2 FA 待测；— |
| libri960 四头 | OTTC 上游 JSD | Wav2Vec2 LV60 / 20ms | 🟡 运行中 18.79/40 | 训练完成后 test | K2 FA 待测；— |
| MFA 标签英语全量 | ctc | Fbank→Linear+LayerNorm→Zapformer / 10ms | ✅ 已完成 10 轮 | ✅ 中途 dev4096 / test192 / K2 FA | ✅ 三个 checkpoint 比较 |
| MFA 标签英语全量 | ottc | Fbank→Linear+LayerNorm→Zapformer / 10ms | ✅ 已完成 10 轮 | ✅ 中途 dev4096 / test192 / K2 FA | ✅ 三个 checkpoint 比较 |
| MFA 标签英语全量 | spottc | Fbank→Linear+LayerNorm→Zapformer / 10ms | ✅ 已完成 10 轮 | ✅ 中途 dev4096 / test192 / K2 FA | ✅ 三个 checkpoint 比较 |
| MFA 标签英语全量 | film_k4 | Fbank→Linear+LayerNorm→Zapformer / 10ms | ✅ 已完成 10 轮 | ✅ 中途 dev4096 / test192 / K2 FA | — |
| MFA 标签英语全量 | ottc_jsd_upstream | Fbank→Linear+LayerNorm→Zapformer / 10ms | ✅ 已完成 10 轮 | ✅ 逐轮 dev；最终 test 待训练 | checkpoint 比较未完成 |
| MFA 标签英语全量 | spottc_jsd_s3t2 | Fbank→Linear+LayerNorm→Zapformer / 10ms | ✅ 已完成 10 轮 | ✅ 逐轮 dev；最终 test 待训练 | checkpoint 比较未完成 |
| MFA 标签英语全量 | spottc_jsd_s3t2_recipe_scratch | Fbank→Linear+LayerNorm→Zapformer / 10ms | ⏳ 正式训练尚未启动；仅 smoke | 最终 test 待训练 | checkpoint 比较未完成 |
| MFA 标签英语全量 | spottc_jsd_s3t2_star_star | Fbank→Linear+LayerNorm→Zapformer / 10ms | 🟡 运行中；step 4745，第 1/10 轮 | ✅ 逐轮 dev；最终 test 待训练 | checkpoint 比较未完成 |

## 本次三个 checkpoint 比较

| 实验 | 状态 | Job |
|---|---|---|
| timit_ctc | ✅ 完整比较已完成 | 3512800.opbs |
| timit_ce | ✅ 完整比较已完成 | 3512801.opbs |
| timit_jsd_upstream | ✅ 完整比较已完成 | 3512802.opbs |
| mfa_ctc | ✅ 完整比较已完成 | 3512785.opbs |
| mfa_ottc | ✅ 完整比较已完成 | 3512786.opbs |
| mfa_spottc | ✅ 完整比较已完成 | 3512787.opbs |
| libri100_ctc | ✅ 完整比较已完成 | 3512788.opbs |
| libri100_ce | ✅ 完整比较已完成 | 3512789.opbs |
| libri100_jsd_upstream | ✅ 完整比较已完成 | 3512790.opbs |

[前端差异与来源](FRONTENDS.md) · [全部结果](RESULTS.md) · [checkpoint 平均结果](../checkpoint_comparison/README.md) · [PBS / W&B](TRACKING.md)。
