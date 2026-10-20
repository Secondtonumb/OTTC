# Zapformer 训练流程核对

检查时间：2026-10-10T14:08:33+09:00。审计对象：3517156 SPOTTC-JSD S3T2，以及它调用的 MFA trainer。

当前训练复用了本地 zipa/icefall Zapformer backbone 和 BatchedRubik，但没有完整执行原 Zapformer recipe 的训练机制。对照是当前运行所用的本地归档分支，不是重新下载的最新 k2-fsa upstream。

## 保存频率

每 1722 个 optimizer step 保存一次，按初始 sampler 6887 steps/epoch 估计，约每0.25轮；间隔跨epoch固定。轮末、进程开始/恢复后第100步和正常限时退出额外保存。保存模型、optimizer混合dtype buffers、scheduler、各rank已消费sampler/RNG、progress和合约。

## 会改变训练结果的差异

| 项目 | 原 Zapformer recipe | 当前 MFA/SPOTTC-JSD | 影响 |
|---|---|---|---|
| 学习率 | InterpCosineLRScheduler，linear_scale=.5，min_factor=.025，按epoch/batch推进 | LambdaLR，warmup_steps=0，base LR=.02恒定 | 缺少全程衰减；以.02起步时参考终值为.0005 |
| 辅助梯度 | aux_loss_scale=.05，step>2000乘.25，再乘AMP scaler scale | forward_encoder固定传0 | 激活幅度/相关性/attention score/gating score惩罚的附加梯度为0 |
| loss与梯度尺度 | batch sum；日志按frames归一化 | DDP全局样本均值；另做clip_grad_norm=5 | 不能只对齐optimizer名字；尤其影响自定义backward辅助梯度和裁剪 |
| SpecAugment | AlternatingSpecAugment：频率与时间mask、time warp，按rank/epoch种子 | 缓存与在线均没有增强 | 缺少recipe增强 |
| AMP | autocast开启时GradScaler，init_scale=1，有恢复/异常检查与scaler保存 | BF16 autocast，无scaler，非有限即停止 | BF16无scaler并非自动错误，但行为与参考不同 |
| 参数平均 | float64累计model_avg，每200 batches更新并保存 | 不维护model_avg | 后处理平均3个checkpoint不能替代累计平均 |
| 输出初始化 | CTC ScaledLinear(initial_scale=.1) | phone/substate输出普通nn.Linear | 初始输出尺度不同 |
| 数据与帧网格 | ASR frontend及Libri增强manifest | 80维packed Fbank→Linear+LayerNorm、10msphone输出 | 任务适配需要保留，但不能称全recipe相同 |

## 已保留的机制与实际适用范围

- BatchedRubik本身保留：矩阵polar-express/Muon方向、逐步变化的动量上限、标量/bias的Adam分支、参数尺度限制、混合dtype state。
- Backbone内的OrthogonalLinear正交梯度惩罚、SequenceNorm/RmsNorm自定义FP32 backward、残差/输入/归一化参数限制以及[-5,5]激活clamp仍在执行。aux_loss_scale=0没有关闭所有内部机制。
- 这份Zapformer源码没有消费batch_count或ScheduledFloat；按当前源码构造tiny同类模型也没有batch_count module。因此当前raw-step赋值是空操作，不能套用其他Zipformer版本的ScheduledFloat/dropout结论。
- multicopy在参考recipe的默认min/max-copies均为1；不能把未支持多副本直接判为漏掉默认启用机制。参考Libri训练manifest的3倍speed perturbation与本实验另有差异。
- RNNT的simple/pruned-loss warmup不适用于SPOTTC-JSD phone objective。compute_projection_overlap调用在该trainer中主要为diagnostics，其单独辅助项已在backbone源码注释，不列作当前失效loss。

## 为什么不能直接打开aux_loss_scale

with_loss的backward给辅助loss固定返回ones，不随主loss上游缩放。CPU检查：主项因子1与.1时，总梯度分别5和4.1，而不是5和.5。当前全局mean归一化、microbatch切分和clip5会影响主/辅项相对尺度；恢复辅助项需要明确按每次forward的缩放与AMP尺度适配。
CorrelationLimiter还有half_batch<=1时不注入辅助梯度的逻辑；改变microbatch大小可能改变它是否生效。AlternatingSpecAugment维护各设备私有Generator，现有通用RNG checkpoint不会自动保存这些Generator。
以上CPU检查证明模块行为，不代表已验证完整GPU/DDP的新训练流程。

## 处理依据

这次只审计和保存证据，运行中的训练与checkpoint未修改。若补齐recipe，应使用有明确配置/源码hash的独立实验和W&B run；从旧checkpoint中途切换无法补回前两轮缺失的训练行为，不能算完整from-scratch recipe对照。优先处理LR、辅助梯度尺度、增强和累计平均，再验证AMP/clip及续训恢复。

[审计细项、配置和源码哈希](ZAPFORMER_TRAINING_AUDIT.json) · [CPU梯度与调度检查](ZAPFORMER_TRICK_CHECKS.json) · [训练进度](TRACKING.md)
