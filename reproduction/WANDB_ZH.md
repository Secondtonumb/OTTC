# W&B 曲线与后续实验记录

项目：https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction

交互式报告：https://wandb.ai/kevingenghaopeng-/ottc-timit-reproduction/reports/TIMIT-fused-phonemes:-loss下降与识别及分段对比--VmlldzoxODA3NDE2NA==

报告包含 CTC、OTTC+CE、原仓库 jsd、标准 JSD、Soft-DTW 五组。
原始训练 loss 每 200 步一条（每组 14 条），评估 loss 每轮一条
（每组 40 条）；补录保留原始 epoch 和更新步，不插值、不生成缺失数据。
各组 cloud history 与本地 Trainer state 的 loss/PER 逐条一致。
这些 run 标记 `logging_origin=backfill`；补录时间不是原训练时间。

报告的 11 个交互图包括：相对初始训练/评估 loss、PER%、学习率、
每个模型单独的原始 train/eval loss，以及 Soft-DTW 第 20–40 轮放大图。
相对 loss 以各自第一条真实记录为 1；CTC/OTTC/Soft-DTW 的 loss
定义和 reduction 不同，原始数值的高低不表示模型性能排序。
原仓库 jsd 与标准 JSD 是两种不同代价，分别展示。

`run_exp.py` 默认同时记录 W&B 和 TensorBoard；后续训练每 50 个更新步
记录 loss，每轮记录 eval loss/PER，同时记录学习率、梯度范数、配置、
数据规模和 run 名称。`reproduction/*.pbs` 已移除旧的禁用配置。
模型目录中的 `wandb_run.json` 保存 run ID 和链接，checkpoint 续训会
连接同一个 run。每次训练结束显式 finish，连续实验各有独立 run。
`evaluate.py` 将音频独立识别和分段 benchmark 指标写入相同 live run summary。

默认项目、账号、模式在 `reproduction/wandb_config.json`，也可通过
`WANDB_PROJECT/WANDB_ENTITY/WANDB_MODE` 或 run_exp 的对应 CLI 参数覆盖。
GPU 节点网络不可用时记录到本地 W&B offline 目录；之后可执行：

```bash
wandb sync /work/gm64/m64000/OTTC/reproduction/results/wandb/wandb/offline-run-具体目录
```

仅上传曲线、配置和评估指标；checkpoint 和原始音频继续保存在本地。
这次添加记录功能没有重新训练已有语音模型。
`check_wandb_tracking.py` 用三步 CPU Trainer 验证完整实时 logging 流程。
`wandb_backfill.py` 可重复运行，已完成的历史 run 不会重复上传曲线。

使用已安装的 wandb 0.21.0；报告 SDK wandb-workspaces 0.4.5
安装在隔离 `reproduction/vendor`，共享环境不变。
