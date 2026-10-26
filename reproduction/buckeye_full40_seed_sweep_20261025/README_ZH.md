# 完整 Buckeye 训练，Buckeye 与 TIMIT 测试

用户确认：四损失 CE / 标准 JSD / AG（原 upstream JSD）/ J-div，各 seed42/43/44，共12组。
每组从原始 wav2vec2-large-lv60 SSL 重新初始化，随机任务头，不复用已有 OTTC 权重。
沿用原始 OTTC transport 和现有散度分支，不重写 AG/JSD 公式。

## 数据

使用已验证的 `buckeye_full40_20261022`，通过 `reproduction.buckeye_dataset` 解析并锁定哈希。
40人、255个原始录音，固定说话人互斥24/8/8划分：

| 划分 | 句数 | 小时 |
|---|---:|---:|
| Buckeye train | 18221 | 12.4217 |
| Buckeye dev | 6417 | 4.1670 |
| Buckeye test | 6839 | 4.1185 |
| TIMIT dev | 400 | 不用于训练或选择 |
| TIMIT test | 192 | 不用于训练或选择 |

完整40人指原始语料覆盖；训练使用既有审计后的切句及过滤结果，不等于所有原始时长均进入训练。
划分沿用 FA-Bench 的说话人集合、IF-MDD legacy 切句，不宣称是 FA-Bench 论文同一切句协议。
保留原有42类词表和映射、edge-sil训练标签及重复标签折叠；TIMIT词表逐项校验一致。
仅 Buckeye dev 参与模型选择。所有 TIMIT 数据只用于最后评估；门禁仅用两句检查评估器能运行，不作性能选择。

## 配方和存储

native10ms，regular，单音素头，无上采样或时间平滑；冻结前六CNN层，训练末层CNN、Transformer及任务头。
40轮，每轮285步，共11400步；每组单GPU物理batch64，尾batch45，不做跨组梯度同步。
AdamW lr2e-4、warmup500、weight_decay0.005、clip1；BF16计算，FP32参数和模型权重。
先检查最长64条/代表64条/尾batch及实际保存权重dtype；不创建FP64模型或平均累加器。
GPU门禁不通过则停止该组，不静默降低batch或改变初始化。门禁用模型不进入正式训练。

每轮按Buckeye dev PER排名，仅保留最佳3份独立FP32模型权重；新优胜权重原子保存后淘汰较差文件，同分保留较早轮次。
另保留最近1份Trainer完整续训状态和best-dev记录；best-dev直接引用三份权重之一，不复制模型。
12组共36份best权重，另有12份续训状态；最终每组生成1份best-3平均模型。
保存替换过程中允许新旧文件短暂共存；所有逐轮指标及淘汰记录保留。
此为用户2026-10-25对此新实验明确指定的精简策略，覆盖之前保留全部40轮/最后10轮的要求。
这套配方在新目录内运行，不修改历史checkpoint、源快照或活动续训契约。
训练标量使用独立W&B run，续训复用run ID；评估指标、预测、音频及报告仅本地。

## 评估与报告

每个seed分别评估：单最佳、best-3（Buckeye dev PER最低三轮等权平均）。不做最后5/10轮平均。
平均仅在同一seed内进行，使用FP32；不是跨seed平均，也不是测试指标平均。
所有候选先完成Buckeye dev FP32评估，按PER选择后才打开Buckeye test、TIMIT dev/test。
报告分别保留各候选以及dev-selected结果，再汇总三个seed的均值和样本标准差。

指标：PER、PFER，text-free及supplied-phone FA各自的20ms边界P/R/F1/R-value，FA MAE。
边界协议：唯一端点、排除句首尾20ms、一对一匹配、冻结IF-MDD R-value；FA使用提供的音素序列，不是音频识别。

- [实验矩阵](EXPERIMENT_MATRIX.json)
- [完成状态和汇总](COMPARISON.json)
- [Markdown 报告](REPORT_ZH.md)

12节点/12GPU regular-g，24小时为作业上限，不是完成承诺；训练预留两小时评估预算。
未完成训练退出75并保存暂停状态，不标为完成。续训须核对原配置和source hashes，不能盲目重复提交。
`python -m reproduction.buckeye_full40_seed_sweep_20261025.submit --submit` 运行CPU门禁、锁定配置并防重复提交。
日志位于 `jobs/<JOBID>/<cost>__seed<seed>/worker.log`；模型及评估位于 `models/<recipe>/`。