# GPU 吞吐审计与调整

更新时间：2026-10-09T07:51:01+09:00。

当前设置偏保守，确有可以省下的计算时间。调整依据是相同有效批次的实际耗时，显存空余不是浪费比例。正式 LibriSpeech 三组各占一张 GH200；MFA 每组沿用恢复契约要求的 8 卡 DDP。目前三组 MFA 加 LibriSpeech 共 27 张正式训练卡，FiLM 尚在排队。

## LibriSpeech：本轮真实单卡测试

GH200 型号标称 120GB，PyTorch 可用容量 95 GiB。固定 64 条真实 train-clean-100 音频，取中位时长附近及最长 64 条两种批次；每配置 4 次，丢弃第 1 次，以下为剩余 3 次平均。BF16、gradient checkpointing、CNN 冻结，包含 collate、复制、四头 loss、backward 和 AdamW 工作；测试 optimizer LR=0，没有更新正式模型。

| 目标 / 阶段 | 批次 | micro=4 秒/步 | micro=8 秒/步 | micro=16 秒/步 | 4→8 吞吐提升 |
|---|---|---:|---:|---:|---:|
| ctc / CTC | 典型 | 1.787 | 1.454 | 1.332 | 22.9% |
| ctc / CTC | 长音频 | 2.227 | 2.096 | 2.279 | 6.3% |
| ce / 冻结前 | 典型 | 1.845 | 1.468 | 1.339 | 25.7% |
| ce / 冻结前 | 长音频 | 2.227 | 2.095 | 2.253 | 6.3% |
| jsd_upstream / 冻结前 | 典型 | 1.854 | 1.481 | 1.345 | 25.2% |
| jsd_upstream / 冻结前 | 长音频 | 2.254 | 2.119 | 2.258 | 6.4% |
| ce / 冻结后 | 典型 | 2.289 | 1.864 | 1.718 | 22.8% |
| ce / 冻结后 | 长音频 | 2.827 | 2.699 | 2.951 | 4.7% |
| jsd_upstream / 冻结后 | 典型 | 2.321 | 1.877 | 1.724 | 23.6% |
| jsd_upstream / 冻结后 | 长音频 | 2.850 | 2.709 | 2.953 | 5.2% |

选择 microbatch=8、gradient accumulation=8，effective batch 仍为 64。16 对典型样本较快，但最长样本略慢。当前 Libri100 继续 microbatch=4 到完成；Libri960 在下一进程启动时自动读取 [策略文件](libri_batch_policy.json)，使用 8。当前执行中的 Python 与 shell 没有被中断。

Libri100 时长 median=13.995s、p99=16.705s、max=24.525s；Libri960 median=13.790s、p99=16.680s、max=29.735s。上表使用 100h 的真实样本，不能直接声称已实测 960h 的最长音频或完整 epoch 加速。后续进程增加实际 step wall time、samples/s、显存与周期 nvidia-smi 采样，并记录在本地和 W&B。

测试使用预取音频，不含正式 data-loader 等待、梯度裁剪和每轮 dev/test 开销，因此表中是批次计算吞吐，不能直接换算整轮节省比例。checkpoint、随机 dropout/SpecAugment 没有逐配置重置；有限重复并非统计显著性测试。冻结阶段单独读取包含 alignment_encoder 的 checkpoint。所有已完成候选 loss 有限、没有 OOM。

首轮验证的 GPU 采样中，显存已分配期间 utilization 中位数为 88%、最大 99%。该统计也含模型加载、回收与切换；不能作为现有正式节点的平均利用率。首轮 PBS 退出记录给出的整段 GPU 平均利用率为 40%、最大 100%，整段约 3m44s，也包含启动和配置切换，保留在调度历史中。正式旧作业当前只有日志中的 step 时间和显存，没有追溯伪造 GPU utilization。

[冻结前吞吐 W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/qcp0hibb) · [冻结后吞吐 W&B](https://wandb.ai/kevingenghaopeng-/ottc-english-multiphoneset/runs/sst4o1dz)。

原始数据：[冻结前结果](throughput_probe/results.json)、[GPU 采样](throughput_probe/gpu.csv)、[冻结后结果](throughput_probe_frozen/results.json)、[冻结后采样](throughput_probe_frozen/gpu.csv)。验证 jobs：3512319.opbs / 3512325.opbs。

## MFA：同模型同数据的历史 8 卡验证

历史 job 3418596 的 batch probe 对固定 sampler 批次执行每候选 10 次真实更新，前 2 次除外，并做最长输入 backward 检查。以下是该历史验证结果，尚不是本轮续训加速实测。

| 目标 | 限额因子 | 原秒/步 | 调整后秒/步 | 吞吐倍数 | 历史 peak GiB |
|---|---:|---:|---:|---:|---:|
| ctc | 4 | 1.469 | 0.751 | 1.96× | 29.44 |
| ottc | 4 | 1.251 | 0.750 | 1.67× | 29.44 |
| spottc | 2 | 1.530 | 0.912 | 1.68× | 22.84 |
| film_k4 | 4 | 0.836 | 0.567 | 1.47× | 21.85 |

历史 FiLM 长音频另测 740–744 sampler batches：因子 4 约 0.611s/步，因子 1 约 0.864s/步；因子 8 约 0.610s/步，额外显存没有对应明显收益。

[历史原始验证结果](/work/gm64/m64000/zipa/mfa_miyabi_20260922/batch_probe_3418596.opbs/results.json) · [FiLM 长样本压力结果](/work/gm64/m64000/zipa/mfa_miyabi_20260922/film740_3418596.opbs/results.json)。

现有正式 CTC / OTTC / SPOTTC 最近 500 次更新平均约 1.32 / 1.43 / 1.51s/步；最大 allocated 约 16.4 / 17.2 / 15.6 GiB。input wait 约 0.000034s，当前日志不支持 data-loader 是主瓶颈。不同目标的 audio/s 和 loss 定义不能直接作模型效率或质量的公平比较。

下一正常退出的 8h 片段由 [续训控制器](mfa_finish_segment.py) 自动提交 [优化脚本](mfa_train_optimized.pbs)。CTC / OTTC 预算乘 4，SPOTTC 乘 2，FiLM 乘 4。FiLM 最终排队 job 为 3512424，四模型推理 gate 已通过；没有取消运行中的训练。CTC 原片段 3507774 已正常保存 step26266、exit0，自动提交优化续训 3512447，正在排队。最新状态以 [进度快照](TRACKING.md) 为准。

[mfa_optimized.py](mfa_optimized.py) 只在进程内调整 microbatch 切分限额；原训练源码、数据、旧 checkpoint 与 migration hashes 保留。logical batch-duration、sampler、optimizer、LR、epochs 和损失定义保持原配置。配置变化及证据 hash 写入 throughput_change.json / W&B，前一段 run_config 归档至 launch_history。新片段增加真实 GPU 采样，下一次 track 应比较该片段吞吐，不能把上面的历史倍数提前写成正式结果。

## 可复现性与验证

改变 microbatch 切分会改变随机 mask、浮点累加顺序；LibriSpeech 的 MFA head 屏蔽样本以每 microbatch 均值处理，切分也会影响该 head 的有效梯度权重。标签、损失定义和 effective 样本批次保持相同，但不能声称逐位相同，也不能保证最终指标完全相同。Libri100 当前比较保持原切分。

已检查策略只作用 Libri960，4/8 切分保留 100h/960h 的有效样本分组与末尾样本覆盖；Python / PBS / shell 语法、源码校验和与 git diff --check 通过。已完成 GPU 候选 finite loss 与 OOM 检查。

[实验进度](TRACKING.md) · [实验 README](README.md)。

## 当前作业是否需要停止

检查：2026-10-09T08:03:59+09:00。当前没有取消作业。

Libri100 约第 32–34 / 40 轮，建议保持当前切分完成，粗估还需约 2–3.5 小时（含评估，可能变化）。Libri960 未启动，下一进程自动使用 microbatch=8。

MFA 当前三组距 7h50m 训练预算只剩约半小时到四十多分钟，建议本片段正常保存退出、下一片段优化；预计约 08:32 / 08:42 / 08:46 JST 加初始化和保存耗时。直接 qdel 只能回到已有 checkpoint，OTTC 检查时最新保存约落后 40 分钟；Libri 首段的 afterok 依赖也可能失效。

排队 FiLM 已使用优化脚本；Libri 续训和评估等待前置成功。排队/held 作业尚未占用运行 GPU，无需为节省 GPU 时间取消重排。
