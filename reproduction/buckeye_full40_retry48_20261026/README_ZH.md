# Buckeye full40：减小实际 batch 后续训

承接失败作业 3706138.opbs 的 12 组、每组 epoch 1 / step 285。
每次前向反向最多 48 条，一次更新处理 64 条（48+16，短尾保持原始平均）；
验证 batch 48。保留原始学习率、优化器、40轮、四损失和三个种子。
模型参数和保存权重为 FP32，计算沿用 BF16。评估只保存在本地。

原始源码、planned_configs、run_config 和已有 source_snapshot 不修改。
runtime.py 显式替换训练入口的 collator 与验证 batch，另存运行时源码与哈希。
initial_checkpoints 通过硬链接保留第285步，原始 tensor 内容不改写。
每组继续使用各自原有 W&B run ID。训练仍用历史 dev-best3 保留及选择合同。

run.pbs 先用 12 个独立进程各占一个节点/GPU，验证最长128条训练音频、
最长48条验证音频、验证后反传、保存和恢复；全部通过才开始正式续训。
ProbeCallback 显式要求保存，避免恢复 trainer_state 后旧 save_steps 导致漏存。
探针权重不用于正式训练。启用 expandable_segments 减少分配碎片。
长作业主动保存暂停后可自动续投；失败退出不会当作成功。

提交：`python -m reproduction.buckeye_full40_retry48_20261026.submit --submit`。
提交记录见 SUBMISSION.json，实际进度见 jobs/ 与原实验 models/。
