# Libri100 训练，TIMIT dev/test 跨库评测：四损失 × 三种子

用户2026-10-22明确要求先提交本组，Buckeye数据范围问题保留，不提交Buckeye。
CE / 标准JSD / AG（旧upstream JSD）/ J-div，各seed42/43/44，共12项。
一次PBS申请12节点，每节点一张GPU、一项独立实验。MPI仅分发任务；
子进程清除分布式rank变量，各自world_size=1，不同步不同损失/seed的梯度。

## 数据与选择

训练只用完整现有LibriSpeech train-clean-100：28538句，100.5865小时。
复用`libri100_singlehead_20261014/data/cache_train`，不重建22GB缓存。
Libri dev-clean+dev-other共5567句，仅用于每轮greedy canonical phone PER选best。
训练/选择只输入音频及标签序列，不输入任何边界时间戳。
沿用已审计ao->aa、zh->sh映射、相邻重复目标折叠，与TIMIT的42类词表ID完全一致。

TIMIT dev400/test192仅用于最终跨库评测，不参与优化、warmup、profile microbatch选择或best选择。
GPU profile另外用固定8句TIMIT dev检查评测器能否完成K2 FA/PFER；这不是模型性能选择。
不加载TIMIT训练集，也不使用任何TIMIT监督checkpoint初始化。
Libri原自动对齐时间戳来源模型未核实，但此处不以其时间戳监督或评分分割。

## 配方与存储

匹配当前TIMIT native10ms系列，而非历史Libri20ms/regfrozen系列：
原始facebook/wav2vec2-large-lv60 SSL、新phone/mass头，前六CNN层冻结，
最后CNN层/Transformer/两任务头可训练，regular，upsample=false，无temporal smoothing。
每项逻辑batch64，40epoch/17840更新，AdamW lr2e-4、warmup500、weight_decay0.005、clip1。
BF16计算，FP32参数和保存权重；microbatch不足64时按完整逻辑batch累计，真实尾batch58。
GPU gate核验原始SSL、梯度、最长64条、代表64条、尾58、最长Libri dev输入、
FP32保存以及8句TIMIT原生K2/PFER。若显存不足仅减microbatch，不改变逻辑batch。
profile模型丢弃，正式训练重新SSL初始化。

只保留Libri dev-PER最佳3份模型权重，加1份最新完整Trainer续训状态，
每10epoch或预算暂停保存完整状态；没有逐epoch权重堆积或checkpoint平均。
每项保留独立本地日志和W&B训练标量；最终TIMIT评测产物与指标仅本地，不上传。

最终独立重建FP32模型，评测单个best-Libri-dev权重：PER、PFER，
text-free与supplied-phone FA各自20ms边界P/R/F1/R-value，以及FA边界MAE。
dev/test分别保存逐句预测、指标与三seed均值/样本标准差；不以test选择checkpoint。

## 提交与审计

`python -m reproduction.libri100_timit_seed_sweep_20261022.submit`只生成矩阵/配置。
加`--submit`实际提交一次regular-g/12节点/24小时任务，回执防止重复提交。
24小时是分配上限，不是完成承诺；预算耗尽前保存可恢复状态，未到40epoch不能标完成。
日志：`jobs/<PBS_JOBID>/<cost>__seed<seed>/worker.log`；权重：`models/<recipe>/`。
汇总：`COMPARISON.json`、`TIMIT_{DEV,TEST}_RESULTS.csv`、
`TIMIT_{DEV,TEST}_THREE_SEED_SUMMARY.csv`。
CPU测试不代替GPU gate，提交/排队也不等同训练完成。
