# 正式训练 GPU 与吞吐快照

检查时间：2026-10-11T17:41:44+09:00。

当前已核对的正式训练共占用 11 张 GPU；排队作业未占用 GPU。

| 模型 | 测量 Job | 状态 | 最近步数 | 平均秒/步 | 最大 allocated GiB | GPU 采样平均利用率 |
|---|---|---|---:|---:|---:|---|
| ctc | 3507774.opbs | 已完成 | 70014 | 0.946 | 39.52 | 无本地连续采样 |
| ottc | 3507775.opbs | 已完成 | 70014 | 0.893 | 39.52 | 无本地连续采样 |
| spottc | 3507776.opbs | 已完成 | 68868 | 1.025 | 24.08 | 无本地连续采样 |
| ottc_jsd_upstream | 3524946.opbs | 已完成 | 70014 | 0.853 | 39.55 | 81.5%（8 个 rank） |
| spottc_jsd_s3t2 | 3517156.opbs | 已完成 | 68868 | 1.171 | 24.04 | 82.0%（8 个 rank） |
| spottc_jsd_s3t2_star_star | 3526510.opbs | 运行中 | 4746 | 0.884 | 24.12 | 80.5%（8 个 rank） |

秒/步按最后 500 条真实更新计算，不含 dev/checkpoint 暂停；音频批次随训练变化，不能把旧片段和本表当作固定输入对照。
GPU 利用率包含当前片段从采样开始至今的初始化、保存和 dev。CTC/CE/SPOTTC 仅有 rank0 CSV，不能代替八卡整体均值；S3T2 有逐卡采样。

3507745 首段正式训练：PBS 记录为三卡 23:05:15，约 69.3 GPU 小时；各卡整段平均利用率 79%、80%、83%。960h 当前续训状态见实验进度表。

[固定输入 GPU 数量及 microbatch 对照](GPU_COST_TEST_RESULTS.md) · [吞吐调整依据](THROUGHPUT.md) · [实验进度](TRACKING.md)

刷新：`python reproduction/english_suite/gpu_runtime_tracking.py`。
