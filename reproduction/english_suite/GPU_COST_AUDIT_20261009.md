# GPU 费用审计

检查时间：2026-10-09 10:00 JST。结论：资源确实用于训练，但目前证据不足以称为成本最优；已有 microbatch 优化，不等于已经优化 GPU 数量。

本轮只读查询作业、日志、配置和历史证据，并新增本记录。没有取消、暂停、重排作业，没有新增付费测速作业或修改训练配置。

## 当前七个作业

| Job ID | 状态 | GPU | 单段时长上限 | 用途与判断 |
|---|---|---:|---:|---|
| 3507745 | RUNNING | 3 | 24h | 三个独立目标各用一张 GPU，先 Libri100 后 Libri960；资源映射合理，持续利用率尚未完整测量 |
| 3507759 | HELD | 1 | 12h | 完成模型的 K2 / text-free 评测；当前未占用运行 GPU，时长上限不是实际耗时 |
| 3507769 | HELD | 3 | 24h | 上一段成功后的续训；当前未占用运行 GPU |
| 3512424 | QUEUED | 8 | 8h | MFA FiLM 从头训练；已选择优化 microbatch，GPU 数量尚无本机成本比较 |
| 3512447 | QUEUED | 8 | 8h | MFA CTC checkpoint 26266 续训；同上 |
| 3512470 | QUEUED | 8 | 8h | MFA OTTC checkpoint 24287 续训；同上 |
| 3512474 | QUEUED | 8 | 8h | MFA SPOTTC checkpoint 22643 续训；同上 |

`qstat` 的 TOKEN 栏不是 GPU utilization。本机这些 small-g GPU 作业中，TOKEN 数值与 GPU 数量乘运行小时相符。四个排队 MFA 片段的申请合计 256 GPU 小时，两个 held 作业合计 84 GPU 小时；这是单段申请量，并非已经发生的费用，也不是全实验总预算。实际货币费用需要计费单价。

3507745 在调度器最近一次采样中已运行 11:51:44，TOKEN=35.6。Libri100 CTC 已完成 17840 步、40 轮并完成内部测试，约 09:56 接上 Libri960；CE / JSD 仍在 Libri100 约第 36.7 / 37.1 轮。当前没有证据表明 CTC 完成后该卡长时间闲置。

## 正式 MFA 整段 GPU 利用率

来源：`qstat -f -H 3507774 3507775 3507776` 的 `resources_used.gpu_avg_util_per_nodes`。每节点一张 GPU、同作业内各节点运行时长相同，以下为八节点数值的算术平均。

| 已完成 Job ID | 目标 | 整段平均利用率 | 节点范围 | walltime | GPU 小时 |
|---|---|---:|---:|---|---:|
| 3507774 | CTC | 68.75% | 63–77% | 07:50:20 | 62.71 |
| 3507775 | OTTC | 72.00% | 69–76% | 07:50:19 | 62.71 |
| 3507776 | SPOTTC | 69.75% | 64–74% | 07:50:19 | 62.71 |

这三段合计约 188.13 GPU 小时，均 exit=0 并保存 checkpoint。不是空跑，但存在优化空间。GPU utilization 不是 FLOPS 利用率，`100%-utilization` 也不能直接解释为可退还或可节省的费用比例。以上是包含初始化、保存等环节的整段指标。

三组下一段已选择扩大 microbatch 切分上限，CTC / OTTC / SPOTTC / FiLM 因子分别为 4 / 4 / 2 / 4。历史 GH200 固定逻辑批次测试吞吐分别提高约 1.96 / 1.67 / 1.68 / 1.47 倍，但下一正式片段尚未开始，不能声称已实现相应全程省费比例。原始证据与限制见 [THROUGHPUT.md](THROUGHPUT.md) 和 `/work/gm64/m64000/zipa/mfa_miyabi_20260922/BATCH_RECOMMENDATIONS.zh.md`。

## LibriSpeech 最新测量

`../multihead_libri960/models/ctc/throughput.jsonl` 前三条记录：50 / 100 / 150 步；microbatch=8、effective batch=64；50 步窗口平均为 1.392 / 1.353 / 1.351 秒每步，约 46–47 条每秒。三个 GPU 瞬时采样为 26 / 24 / 24%，显存使用约 11–13 GiB。

采样发生在 loss logging 回调中，可能偏向步末阶段，只有三个瞬时值，不能据此报告整段平均 GPU 利用率或断言浪费了 75% 费用。需要连续、独立于训练步的采样及同一有效批次的端到端耗时比较。通过 SSH 直接读取运行节点本轮被认证拒绝；没有改动 SSH 配置或凭据。旧 Libri100 进程未加载新增 telemetry，不能补造其历史利用率。

已有 GH200 microbatch=4/8/16 测试支持选择 8：典型批次相对 4 吞吐提高约 23–26%，长输入约 5–6%；16 对长输入略慢。测试未包含全部正式加载、评测和梯度裁剪成本，也不覆盖完整 Libri960 epoch，因此不等于整轮省费比例。

## 八卡依据及缺失证据

这里的 MFA 是 Zapformer 神经网络训练，不是经典 MFA CPU 对齐命令。`mfa_worker.sh` 启动八节点 true DDP；原 trainer 明确 `assert world==8`。`step_checkpoint.load` 要求恢复 GPU 数量与 checkpoint 相同，并恢复每 rank 的 sampler / RNG 状态。CTC / OTTC / SPOTTC 不能只改 PBS 为四卡就继续；FiLM 尚未正式训练，但现有入口也固定八卡。

找到了历史 A100 40GB、单机四卡对八卡测试：扩展效率约 97–100%。来源：`/work/gm64/m64000/zipa/mfa_miyabi_20260922/zipa/experiments/mfa_zapformer/validation/SCALING_REPORT.md`。该测试保持每卡时长预算，因此全局 batch 随卡数翻倍；不是当前 Miyabi 跨节点 GH200、固定全局 batch 的成本比较。八卡恢复契约解释了当前配置，却不能证明八卡经济最优。

合理下一步是固定全局有效样本 / 音频量，比较本机较少 GPU 的端到端 GPU 小时，并验证数据覆盖、loss 归一化及 checkpoint 迁移。四卡同一有效 batch 的耗时若小于八卡的两倍，GPU 时间成本才下降；不能只比较每步快慢。迁移方案需先验证，不能简单跳过恢复检查。

当前 MFA telemetry 脚本只在主节点采样，下一段还应补全八节点连续采样，才能判断 rank 不均衡和通信等待。

## 总费用边界

`finish_segment.py` 和 `mfa_finish_segment.py` 在未达到 LibriSpeech 40 轮 / MFA 10 轮时自动提交下一段，没有整个实验的 TOKEN 预算上限。当前队列单段 TOKEN 的和不能用于承诺实验总费用。

建议将后续资源决策以固定工作量的 GPU 小时和明确总预算为依据；优先完成当前 microbatch 优化的正式测量，再决定是否值得迁移到更少 GPU。运行中 LibriSpeech 应通过正常保存退出进行调整，避免丢失已有更新或使 `afterok` 依赖失效。
