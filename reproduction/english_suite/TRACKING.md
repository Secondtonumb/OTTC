# 英语实验进度

检查时间：2026-10-11T17:41:43+09:00。这是实际查询的快照。

LibriSpeech 全量数据审计与三组官方 LV60 GPU smoke 已通过。
MFA 四组真实更新与 CTC / FiLM 保存后恢复验证：PASS。
Libri100 三组已完成 40 轮；Libri960 是否运行以续训 Job 和保存标记为准，首段 exit0 不代表 40 轮全部完成。

| 作业 | Job ID | 当前状态 | 实际 / 预计启动（JST） |
|---|---|---|---|
| LibriSpeech 第一段 | 3507745.opbs | 已退出（exit=0） | 实际：Thu Oct  8 22:07:46 2026 |
| LibriSpeech 第二段 | 3507769.opbs | 运行中 | 实际：Sun Oct 11 04:44:06 2026 |
| LibriSpeech K2 / text-free 评估 | 3507759.opbs | 已退出（exit=0） | 实际：Sat Oct 10 22:49:06 2026 |
| MFA ctc | 3507774.opbs | 已退出（exit=0） | 实际：Fri Oct  9 00:42:22 2026 |
| MFA ottc | 3507775.opbs | 已退出（exit=0） | 实际：Fri Oct  9 00:52:08 2026 |
| MFA spottc | 3507776.opbs | 已退出（exit=0） | 实际：Fri Oct  9 00:56:21 2026 |
| MFA film_k4 | 3512424.opbs | 已退出（exit=0） | 实际：Sat Oct 10 23:10:40 2026 |
| ottc_jsd_upstream | 3515322.opbs | 已退出（exit=0） | 实际：Sat Oct 10 00:18:29 2026 |
| ottc_jsd_upstream latest_training_job | 3524946.opbs | 已退出（exit=0） | 实际：Sun Oct 11 06:40:39 2026 |
| spottc_jsd_s3t2 | 3517156.opbs | 已退出（exit=0） | 实际：Sat Oct 10 10:24:34 2026 |
| spottc_jsd_s3t2 evaluation_job | 3525524.opbs | 已退出（exit=0） | 实际：Sun Oct 11 10:04:33 2026 |
| spottc_jsd_s3t2_recipe_scratch | 3521280.opbs | 已退出（exit=未知） | — |
| spottc_jsd_s3t2_star_star | 3526510.opbs | 运行中 | 实际：Sun Oct 11 16:23:19 2026 |
| mfa_models/ottc/segment_status.json evaluation_job | 3521908.opbs | 已退出（exit=0） | 实际：Sat Oct 10 18:24:00 2026 |
| mfa_models/ctc/segment_status.json evaluation_job | 3521464.opbs | 已退出（exit=0） | 实际：Sat Oct 10 15:56:29 2026 |
| mfa_models/spottc/segment_status.json evaluation_job | 3522171.opbs | 已退出（exit=0） | 实际：Sat Oct 10 22:49:07 2026 |
| mfa_models/film_k4/segment_status.json continuation_job | 3526846.opbs | 排队 | 预计：Sun Oct 11 22:55:42 2026 |
| 吞吐验证 libri_probe_job | 3512319.opbs | 已退出（exit=0） | 实际：Fri Oct  9 07:42:15 2026 |
| 吞吐验证 libri_frozen_probe_job | 3512325.opbs | 已退出（exit=0） | 实际：Fri Oct  9 07:47:23 2026 |
| 评测补齐 mfa_prepare_job | 3512374.opbs | 已退出（exit=0） | 实际：Fri Oct  9 08:20:31 2026 |
| 评测补齐 mfa_current_dev_test_job | 3512419.opbs | 已退出（exit=0） | 实际：Fri Oct  9 08:25:50 2026 |
| 评测补齐 mfa_inference_gate_job | 3512417.opbs | 已退出（exit=0） | 实际：Fri Oct  9 08:24:25 2026 |
| MFA ctc current dev/test | 3512419.opbs | 已退出（exit=0） | 实际：Fri Oct  9 08:25:50 2026 |
| MFA ottc current dev/test | 3512434.opbs | 已退出（exit=0） | 实际：Fri Oct  9 08:26:21 2026 |
| MFA spottc current dev/test | 3512435.opbs | 已退出（exit=0） | 实际：Fri Oct  9 08:26:22 2026 |
| 三个 checkpoint 比较 timit_ctc | 3512800.opbs | 已退出（exit=0） | 实际：Fri Oct  9 09:53:24 2026 |
| 三个 checkpoint 比较 timit_ce | 3512801.opbs | 已退出（exit=0） | 实际：Fri Oct  9 09:53:26 2026 |
| 三个 checkpoint 比较 timit_jsd_upstream | 3512802.opbs | 已退出（exit=0） | 实际：Fri Oct  9 09:53:27 2026 |
| 三个 checkpoint 比较 mfa_ctc | 3512785.opbs | 已退出（exit=0） | 实际：Fri Oct  9 09:54:13 2026 |
| 三个 checkpoint 比较 mfa_ottc | 3512786.opbs | 已退出（exit=0） | 实际：Fri Oct  9 09:54:14 2026 |
| 三个 checkpoint 比较 mfa_spottc | 3512787.opbs | 已退出（exit=1） | 实际：Fri Oct  9 09:54:13 2026 |
| 三个 checkpoint 比较 libri100_ctc | 3512788.opbs | 已退出（exit=0） | 实际：Sat Oct 10 06:14:25 2026 |
| 三个 checkpoint 比较 libri100_ce | 3512789.opbs | 已退出（exit=0） | 实际：Sat Oct 10 03:44:54 2026 |
| 三个 checkpoint 比较 libri100_jsd_upstream | 3512790.opbs | 已退出（exit=0） | 实际：Sat Oct 10 06:28:38 2026 |
| Zapformer 上游 JSD GPU 验证 | 3512964.opbs | 已退出（exit=0） | 实际：Fri Oct  9 10:31:48 2026 |
| Libri GPU 吞吐验证 | 3512871.opbs | 已退出（exit=0） | 实际：Fri Oct  9 10:15:45 2026 |
| MFA GPU 吞吐验证 | 3512870.opbs | 已退出（exit=1） | 实际：Fri Oct  9 10:26:50 2026 |
| Libri100 JSD FA-bench | 3515510.opbs | 已退出（exit=0） | 实际：Fri Oct  9 17:18:42 2026 |
| Zapformer recipe S3T2 单卡验证 | 3521267.opbs | 已退出（exit=1） | 实际：Sat Oct 10 14:40:41 2026 |
| Zapformer recipe S3T2 八卡验证 | 3521279.opbs | 已退出（exit=未知） | — |
| Zapformer recipe GPU恢复诊断 | 3527782.opbs | 排队 | 预计：Sun Oct 11 17:45:59 2026 |
| GPU 成本短测 mfa_initial | 3512870.opbs | 已退出（exit=1） | 实际：Fri Oct  9 10:26:50 2026 |
| GPU 成本短测 mfa_retry | 3513254.opbs | 已退出（exit=0） | 实际：Fri Oct  9 11:49:25 2026 |
| GPU 成本短测 libri | 3512871.opbs | 已退出（exit=0） | 实际：Fri Oct  9 10:15:45 2026 |

预计启动时间会随调度变化。等待前置作业不是训练停滞；正式 run 启动后才有正式 loss 曲线。

## 正式训练进度

| 模型 | 最近实际步数 | 轮次 | 最近记录 train loss | W&B |
|---|---:|---|---:|---|
| libri100/ctc | 17840 | ✅ 已完成 40.0 轮 | 0.0582 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/jbhs9cm0) |
| libri100/ce | 17840 | ✅ 已完成 40.0 轮 | 0.3274 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/myw8w964) |
| libri100/jsd_upstream | 17840 | ✅ 已完成 40.0 轮 | 0.9764 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/459q6sop) |
| libri960/ctc | 62694 | 运行中；14.26 / 40（最近 loss 记录） | 0.0586 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/rrkrav4i) |
| libri960/ce | 57400 | 运行中；13.06 / 40（最近 loss 记录） | 0.4791 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/sxmpy2g1) |
| libri960/jsd_upstream | 57206 | 运行中；13.02 / 40（最近 loss 记录） | 1.5420 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/11ka2qzu) |
| mfa/ctc | 70014 | ✅ 已完成 10 轮 | 21.2724 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/n7rxjnpf) |
| mfa/ottc | 70014 | ✅ 已完成 10 轮 | 1.2528 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/wi9ghjar) |
| mfa/spottc | 68868 | ✅ 已完成 10 轮 | 3.8821 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/k1rqy7lb) |
| mfa/film_k4 | 102406 | 已保存，续训排队；第 10 / 10 轮（epoch 从 0 计） | 17.9436 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/6jxqmwvm) |
| mfa/ottc_jsd_upstream | 70014 | ✅ 已完成 10 轮 | 3.2876 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/egx42sf3) |
| mfa/spottc_jsd_s3t2 | 68868 | ✅ 已完成 10 轮 | 5.7695 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/r6sjzi4r) |
| mfa/spottc_jsd_s3t2_recipe_scratch | — | 验证失败，正式训练未启动；尚未启动 | — | — |
| mfa/spottc_jsd_s3t2_star_star | 4745 | 运行中；第 1 / 10 轮（epoch 从 0 计） | 6.6745 | [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/5v25b41z) |

步数来自最新日志，loss 按原记录间隔展示；checkpoint 可能早于最新训练步。原始 loss 定义不同，不直接比较量级。

## LibriSpeech 最近 dev（原始 OTTC ARPAbet head，text-free）

| 模型 | Dev epoch | PER % | F1@20ms % |
|---|---:|---:|---:|
| libri100/ctc | 40.00 | 3.75 | 55.54 |
| libri100/ce | 40.00 | 4.42 | 72.13 |
| libri100/jsd_upstream | 40.00 | 4.64 | 74.20 |
| libri960/ctc | 14.00 | 2.91 | 48.23 |
| libri960/ce | 13.00 | 6.08 | 69.19 |
| libri960/jsd_upstream | 13.00 | 5.94 | 72.85 |

中途 dev 结果来自不同 epoch，不能当作最终公平比较；LibriSpeech 边界参考是自动对齐。Libri100 checkpoint 比较和独立 JSD FA-bench 已有结果，Libri960 最终 K2 FA 仍等待训练完成。

## MFA 外部 TIMIT dev（text-free）

| 模型 | checkpoint step | 阶段 | PER % | Precision@20ms % | Recall@20ms % | F1@20ms % |
|---|---:|---|---:|---:|---:|---:|
| ctc | 70014 | 正式逐轮 dev | 41.32 | 32.98 | 58.96 | 42.30 |
| ottc | 70014 | 正式逐轮 dev | 56.41 | 47.36 | 52.01 | 49.57 |
| spottc | 68868 | 正式逐轮 dev | 44.47 | 35.51 | 67.97 | 46.65 |
| film_k4 | 98470 | 正式逐轮 dev | 50.40 | 35.25 | 70.23 | 46.94 |
| ottc_jsd_upstream | 70014 | 正式逐轮 dev | 52.14 | 48.76 | 50.81 | 49.77 |
| spottc_jsd_s3t2 | 68868 | 正式逐轮 dev | 50.77 | 26.54 | 62.13 | 37.19 |
| spottc_jsd_s3t2_recipe_scratch | — | 尚无完整 dev 指标 | — | — | — | — |
| spottc_jsd_s3t2_star_star | 0 | 正式逐轮 dev | 100.00 | 15.33 | 96.32 | 26.45 |

恢复时及逐轮完整 dev logging 已在当前续训生效。历史缺失的逐轮指标不补造。

[实验完成状态总表](EXPERIMENT_STATUS.md) · [前端区别](FRONTENDS.md) · [全部结果](RESULTS.md) · [三个 checkpoint 的比较](../checkpoint_comparison/README.md)。

## 验证运行 W&B（不代表正式长训）

Zapformer 上游 JSD GPU 验证：PASS；完整模型 loss/梯度与原 OTTC 归档实现一致。
独立 JSD 长训已启动，当前状态见上表；[实现与验证](mfa_jsd/README.md)。

- Zapformer JSD smoke：[W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/n6x02go0)
- validation/ce: [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/incke2ug)
- validation/ctc: [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/w4sb0w68)
- validation/jsd_upstream: [W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/dw2dbax3)
- mfa_validation/3507763.opbs/ctc: [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/iiuknra8)
- mfa_validation/3507763.opbs/ctc_reload: [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/1x5zza9u)
- mfa_validation/3507763.opbs/film_k4: [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/i28l47l0)
- mfa_validation/3507763.opbs/film_k4_reload: [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/25hyfll0)
- mfa_validation/3507763.opbs/ottc: [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/7gefy57o)
- mfa_validation/3507763.opbs/spottc: [W&B](https://wandb.ai/kevingenghaopeng-/ottc-mfa-english/runs/coi30x4p)

## GPU 吞吐调整

[实测协议、吞吐结果与下一段生效配置](THROUGHPUT.md)。

[GPU 成本短测结果与连续利用率](GPU_COST_TEST_RESULTS.md)。

[当前正式续训吞吐与 GPU 采样](GPU_RUNTIME_TRACKING.md)。

Libri100 已用 microbatch=4 完成；Libri960 续训策略为 microbatch=8、effective batch=64。
MFA 当前 CTC/OTTC/SPOTTC 续训已分别启用限额因子 4/4/2；排队 FiLM 为 4。

刷新：`python reproduction/english_suite/track.py`。

[完整实验配置与调度记录](README.md)。
