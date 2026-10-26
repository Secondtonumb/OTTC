# TIMIT 单卡正式对照与三个新seed

优先J-div seed42，再补CE、标准JSD、AG、J-div各seed43/44/45，共13个新训练。
CE/标准JSD/AG的历史seed42已完成，直接复用，不重训。

全部单卡、batch64、40epoch/2920步。匹配历史
`timit_ssl_native10_20261019/models/wav2vec2__timit__10ms__mono__*`：
原始wav2vec2-large-lv60 SSL、新任务头、native10ms、前六层CNN冻结，
最后一层CNN/Transformer/两个头可训练，regular forward，无temporal smoothing。
LR2e-4、warmup500、weight_decay0.005、clip1、BF16计算、FP32参数和权重。
三种新seed只改变初始化/采样随机种子；训练数据4620、dev400、test192和评分器固定。

存储仅保留dev PER最佳3份权重，以及1份最新完整Trainer续训状态。
`best_dev.pt`仅为最佳权重的符号链接。每轮dev评分选best；每10轮更新完整续训状态，
预算暂停时也保存。无逐epoch权重、无final/averaged权重副本、无profile权重残留。

最终同一best-dev单模型在FP32下评分：PER、PanPhon PFER，
以及text-free与supplied-phone forced alignment两套segmentation P/R/F1/R-value。
FA另报边界MAE。PFER仅用音频解码假设，移除sil/q，dx->d，与PER归一化一致；
IPA双元音展开，采用已有已验证的PanPhon0.22.2特征距离后端和feature table。
先dev选定权重再test，不使用test挑模型。

为匹配旧结果，使用历史`benchmark_best_dev_test.json`，不混入旧final5/final10平均。
最终`COMPARISON.json`和`COMPARISON.csv`由各job完成后自动更新，
主表`MERGED_COMMON_SEEDS.csv`统一合并历史与新实验的seed42/43/44，报告mean和sample std。
`MERGED_ALL_SEEDS.csv`仅为所有已完成seed的补充，缺失seed及不同样本数明确标注。
JSON同时保留原新seed43/44/45统计；未完成条目保持pending，不伪造结果。
历史PFER可从原先保存的预测重新计分，输出到本实验目录，不改历史文件。

提交：`qsub -v SEED_RECIPE=jdiv__seed42 -o 绝对日志路径 run.pbs`。
使用regular-g单节点单GPU，walltime2小时；训练标量/W&B曲线与本地证据均保留，
评估预测/音频/表格不上传。profile严格验证物理batch64、最长输入、完整SSL初始化、
梯度、FP32保存以及12条尾batch。训练模型重新初始化，不继承profile更新。
