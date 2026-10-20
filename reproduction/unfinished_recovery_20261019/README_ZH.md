# 未完成实验接续记录

核验时间（UTC）：2026-10-19T02:38:20.580762+00:00。用户要求每个作业和自动接续均使用 1 小时。

| 实验 | PBS 作业 | 状态 | 时限 |
|---|---|---|---|
| wav2vec2__timit__10ms__s3t2ss__ce | 3586599.opbs | R | 01:00:00 |
| wav2vec2__timit__10ms__s3t2ss__jsd_standard | 3586600.opbs | R | 01:00:00 |
| wav2vec2__timit__20ms__mono__crottc_jsd_upstream | 3586613.opbs | R | 01:00:00 |
| wav2vec2__timit__20ms__mono__crottc_ce | 3586614.opbs | R | 01:00:00 |
| wav2vec2__timit__20ms__mono__crottc_jsd_standard | 3586615.opbs | Q | 01:00:00 |

两组 S3T2** 的 epoch 1 / step 73 重放与原权重逐张量完全一致（最大差异 0），已重建 optimizer/scheduler/RNG 并进入原训练器接续；原始 epoch 权重与源码保持不变，截断 checkpoint 移至 recovery_preserved 保留。重放不向原 W&B 记录重复训练点；正式接续沿用 26ll7g6i / xsmxmy6d。

CROTTC：SpeechBrain 首次导入重新启用 TF32；验证在增强器导入后关闭 TF32，原始 loss/梯度阈值不变。当前源码的完整 CPU/原始 SSL 与 CUDA 验证均通过；CUDA exact-resume 差异 0。训练仍用原 BF16 配置，三组各自通过真实数据 profile 后才开始生产训练。

每个作业前 40 分钟用于 profile/恢复/训练，后 20 分钟留给保存和评测；未完成时以 1 小时作业接续。每轮权重、best/latest、final5/final10 平均及 W&B 策略保留。四组已完成实验不重跑。

恢复证据：各输出目录的 RECOVERY_PASS.json；数值诊断 CROTTC_NUMERICS.json；完整核验 STATUS.json；历史与新作业映射在两套 dispatch.json。
