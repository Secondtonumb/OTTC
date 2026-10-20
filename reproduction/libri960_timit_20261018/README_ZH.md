# Libri960 加训前后的 TIMIT 零样本评测

用户于 2026-10-18 要求测 TIMIT。本目录独立保存评测代码和产物，不修改已完成的训练代码、权重或旧评测。

三种目标分别是原上游 JSD、CE 和现有 standard JSD。每组比较自己的 Libri100 epoch40 父 checkpoint、Libri dev 已选 best-dev、Libri960 final 和新增五个完整 epoch 的平均。不用 TIMIT dev/test 重新挑选 checkpoint，不在 TIMIT 上微调。

数据复用 `reproduction/multihead_timit/data` 中的 dev400 和 core-test192。core-test192 的音频路径与人工参考逐句匹配历史单头 JSD 的评分文件。参考保留 sil 和原始未取整时间，closure 作 sil，q 去除并保留间隙；识别评分去 sil/q，dx→d。给定音素 FA 使用包含 sil 的精确参考序列，不输入参考时间。

无文本模式使用音频 argmax 及固定20ms时钟；K2 FA 复用冻结 IF-MDD 实现及 duration/frame_count 时钟、blank previous。评分复用之前的边界抽取、一对一匹配、PER 与冻结 R-value 函数，记录源哈希。报告10/20/30ms结果；20ms为主表。

完整评测覆盖 12 个模型配置 × 592 句，共 7,104 次。每句另跑 labelled forward 记录原始 OTTC loss；预测只取 audio-only forward，验证提供标签不改变 logits。不同目标的 loss 不直接比较。FA失败保留在全集边界计数中；时间误差用每目标四变体共同成功子集。

已在 Miyabi 申请一节点 GH200、20分钟 interactive allocation，job `3578963.opbs`，节点 `mg0010`。先验证最长、最短和中等长度语音，再执行全量。每个完整变体可独立复用，避免中断后重做已经完成的评测。最终校验每句覆盖、唯一ID、manifest与预测哈希、checkpoint身份，并将报告和逐句结果上传 W&B；网络不通则保留 offline 日志。

在 GPU allocation 内运行：

```bash
bash reproduction/libri960_timit_20261018/run.sh
```

`COMPARE_COMPLETE.json` 表示全部本地结果验证通过；`WANDB_RUNS.json` 记录发布结果。具体结果见 `REPORT_ZH.md`、`RESULTS.json` 和 `evaluation/<objective>/<variant>/`。
