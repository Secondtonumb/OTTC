# Buckeye 四损失 × 三种子

CE / 标准JSD / AG（旧upstream JSD）/ J-div，各seed42/43/44。
一次PBS申请12节点，每节点一张GPU、一项独立实验。MPI只负责分发，
训练子进程清除rank/world-size环境，绝不在不同损失或seed间同步梯度。

沿用`buckeye_singlehead_20261014/data`的原始人工标注19-speaker子集，
不是完整40人Buckeye。固定speaker-disjoint 15/2/2划分：
train10927句7.3338小时，dev1524句1.1100小时，test1840句0.8699小时。
dev为s04/s14，test为s21/s22；划分种子固定为旧seed0，不随训练seed变化。
训练标签含原有edge-sil、ao->aa并折叠相邻重复标签；评测保留人工speech-phone区间。

匹配本次TIMIT而不是历史Buckeye20ms/regfrozen配方：原始wav2vec2-large-lv60 SSL，
新任务头，native10ms，前六CNN层冻结，最后一层CNN/Transformer/两任务头可训练，
regular、无temporal smoothing。每项独立逻辑batch64、40epoch/6840更新，
AdamW lr2e-4、warmup500、weight_decay0.005、clip1，BF16计算、FP32参数及保存权重。
先检查最长64条、代表64条及真实尾batch47，必要时减小物理microbatch，
按实际逻辑batch大小归一化一次梯度更新，不把64分摊到12项实验。
GPU profile会核验SSL加载、冻结/训练参数、梯度、显存和保存dtype；profile权重不进入训练。

每轮dev PER选best，仅保留最佳3份权重和1份最新完整续训状态；每10epoch保存续训状态。
最终独立FP32 best-dev模型报告dev/test PER、PFER，text-free与supplied-phone FA
分开报告20ms边界P/R/F1/R-value，FA另报MAE。不能把FA目标序列当音频识别结果。
历史Buckeye20ms/final40模型不能并入此组seed均值。

`submit.py`默认仅生成矩阵和配置；`--submit`提交一次12节点regular-g、6小时任务，
保留PBS回执且防重复提交。训练标量启用W&B，评测预测/指标/产物仅本地保存。
到时限前保留完整checkpoint，不把暂停标成完成；同配置可继续训练。
输出在`models/<cost>__seed<seed>`，rank日志在`jobs/<PBS_JOBID>/<recipe>/worker.log`。
只有12项的BENCHMARK_COMPLETE均存在，COMPARISON才标记complete。
