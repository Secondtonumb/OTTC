# GPU 成本短测计划与作业记录

2026-10-09 用户授权进行测试。两个隔离短作业于 10:11:53 JST 提交，使用 debug-g，每个最多 20 分钟，不自动重试或续投。总申请上限：8×20/60 + 1×20/60 = 3 GPU 小时。

| 作业 | Job ID | 资源 | 提交后预计启动 | 输出 |
|---|---|---|---|---|
| MFA 固定全局 batch 卡数比较 | 3512870.opbs | 8 GPU / 20min | 2026-10-09 11:40:41 JST | gpu_cost_probe_3512870.opbs |
| Libri960 单卡配置比较 | 3512871.opbs | 1 GPU / 20min | 2026-10-09 11:27:34 JST | gpu_cost_libri_3512871.opbs |

预计时间可能变化；截至提交后检查仍排队，没有结果。作业运行时每卡连续 1 秒采样，日志、telemetry、模型副本、源码快照及 W&B 均保留在各自输出目录。没有取消其他作业或修改正式训练配置。

尝试将两个短作业迁到 regular-g 回填，但本环境禁止执行 `/opt/pbs/bin/qmove`，SSH 访问登录节点也未通过 host-key 校验，因此仍保留 debug-g 原作业，无重复提交。最后查询预计启动时间同上。

## MFA 比较合同

- 四个目标：CTC / OTTC / SPOTTC / FiLM K4。
- 相同全量语料 manifest 和已记录真实 sampler fixtures，每步八个 virtual rank 的全局样本 ID 保持不变，检查覆盖和无重复。
- 卡数 8→4→2→1 和 1→2→4→8 两次独立重复；每档 10 次真实更新，前 2 次预热。
- 使用正式下一段优化 microbatch 因子 4/4/2/4，BF16、原模型、BatchedRubik 优化器、LR=.02、梯度裁剪。
- CTC / OTTC / SPOTTC 加载隔离 checkpoint 副本（26266 / 24287 / 22643），保留混合 dtype 优化器 buffers；FiLM 使用相同 seed 的初始模型。
- 每档先独立 backward 容量检查、最长真实输入检查；通过后进入 DDP 计时。有效 loss 按全局样本数归一化。
- 端到端批次计时含预取等待、复制、forward/backward、梯度同步、裁剪、optimizer；排除初始化、预热、每步 telemetry 归并和同步起点。
- 指标：秒/更新、音频秒/墙钟秒、GPU 秒/音频小时、逐卡连续 GPU utilization、峰值显存、finite loss/gradient。
- 原训练源码明确要求八卡恢复；本测试加载模型/优化器做独立更新，不恢复 sampler/RNG、不写回正式 checkpoint，不构成正式少卡续训迁移验证。
- 同一全局数据量不代表相同 mask 或逐位相同梯度；较少卡重新切分 microbatch，可能影响随机增广和浮点累计。

本次为了共用同一批节点申请八卡，较少卡阶段其他卡仍属于本次 allocation 消耗。候选成本按参与计算的卡数归一化，实际测试计费以 PBS 整段八卡时间为准。

## LibriSpeech 比较合同

- 单卡，CTC / CE / JSD 三个已有模型的隔离副本。
- 固定真实 Libri960 典型 64 条与最长 64 条，有效 batch=64。
- microbatch=8/16/32，分别比较 gradient checkpointing 开/关。
- 每档从相同模型参数开始，共 7 次真实更新，首步预热；BF16、冻结 CNN、原 AdamW LR 和梯度裁剪。
- CE/JSD 使用已有冻结对齐分支的 checkpoint，按冻结阶段恢复参数冻结；结论仅适用于该阶段。
- 音频预读取，计时包含 collate、复制、四头 loss、backward、裁剪和 AdamW；不代表整轮 IO/dev/test 成本。
- 持续 1 秒 GPU 采样与每档时间窗关联，避免只在日志回调中采样的偏差。
- 切分变化可能影响 masked-head 权重，关闭重计算也可能改变随机数路径；没有收敛质量保证。

## 文件与验证

入口：`gpu_cost_mfa.pbs` / `gpu_cost_worker.sh` / `benchmark_mfa_gpu_cost.py` 和 `gpu_cost_libri.pbs` / `gpu_cost_plan.py` / `benchmark_libri_gpu_cost.py`。

汇总：`summarize_gpu_cost.py`；每个完成作业生成自己的 `summary.md`，也可刷新 [汇总报告](GPU_COST_TEST_RESULTS.md)：

```bash
/work/gm64/m64000/miniforge3/envs/sb_k2_cuda/bin/python reproduction/english_suite/summarize_gpu_cost.py
```

提交前 shell/Python 语法、正式 runtime hashes 和 `git diff --check` 通过。最终结果需要 `BENCHMARK_COMPLETE.json` 和 PBS 退出状态；当前未声称 GPU 测试通过。

## 2026-10-09 11:40 JST 跟踪

LibriSpeech 3512871 于 10:15:45 开始，14m30s、exit0 完成 36 个候选；实际 0.2417 GPU 小时。基线和关闭重计算 microbatch8 的典型/最长批次均通过；关闭重计算 microbatch16/32 的最长批次 OOM。固定计算区间连续利用率约 93–100%，整作业调度器平均为 60%（含读取、模型重置和候选切换）。

MFA 3512870 于 10:26:50 开始，12 秒、exit1；实际 0.0267 GPU 小时。第一处错误是 Gloo connectFullMesh，主节点到自身路由选择 lo，未执行训练。保留失败日志；修复网卡选择和 IPv4，拒绝 lo 后补投 **3513254.opbs**，仍为八卡/20min。补投上限加前两作业实际消耗合计 2.935 GPU 小时，仍在原定 3 GPU 小时内。不会自动再次补投。

结果与状态源：[GPU_COST_JOBS.json](GPU_COST_JOBS.json)、[GPU_COST_TEST_RESULTS.md](GPU_COST_TEST_RESULTS.md)。正式训练配置没有改变。
