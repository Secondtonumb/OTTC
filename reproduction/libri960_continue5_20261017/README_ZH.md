# Libri100 三种 OTTC → 完整 Libri960 加训5轮

CE、原上游JSD、standard JSD分别从已经完成的Libri100单头epoch40/step17840开始，保留对应AdamW moments和已冻结的regfrozen对齐分支，不从预训练重新训练。原checkpoint、旧训练代码、W&B运行及评测保持原样。新增阶段各自建立W&B运行，记录父运行ID、权重哈希和数据来源；同阶段中断恢复沿用新运行的ID。

数据为完整train-clean-100 + train-clean-360 + train-other-500，共281,191句、960.87小时、2,338位说话人，包含原train100。保持官方dev/test说话人隔离。没有裁剪或按时长删除样本，最长29.735秒。训练只使用既有42类ottc_arpabet音素序列、音频及音频长度；不读取参考时间戳。compact Arrow不复制960小时音频，按需读取FLAC并执行与原Libri100缓存逐样本相同的normalization，已用旧缓存验证。

全局batch64、BF16 autocast及FP32主权重、seed42、AdamW weight decay0.005。学习率统一2e-5，为原训练的十分之一；继承优化器moments，开启新的5轮线性学习率调度、warmup500。原40轮调度已经到零，因此不能直接沿用其零学习率。每轮4,394 optimizer step，5轮共21,970 step，累计39,810 step。原模型的CNN和对齐分支继续冻结，phone Transformer继续微调；不重新执行30轮冻结切换，不修改CE、原上游reverse-KL JSD或现有standard JSD公式。

按用户要求，每epoch约1/3、2/3及结束存一次完整checkpoint，具体第一轮为stage step1465、2930、4394，后续每4,394步重复，共15个。完整checkpoint保留模型、optimizer、scheduler、Trainer及RNG状态，不轮转删除。allocation预算耗尽时另保存可恢复latest，供同入口续跑。每个完成epoch的模型权重还原子写入epoch_weights/epoch-001..005，独立于latest状态，附epoch/step、父模型/配置身份和SHA256。不将半个epoch记为完整epoch。

epoch最佳选择使用与父训练相同的BF16折叠训练目标dev去sil PER，与最终FP32参考评分分开。最佳模型只由dev去sil PER选择，候选包含原Libri100 epoch40基线，以便保留加训前更好的模型。每个新epoch结束记录原始dev loss、识别PER及text-free边界P/R/F1/R-value。完成后分别评测final、best_dev、average_last5。last5只平均新增的五个完整epoch，记录成员路径、epoch、step和哈希；只授权新增5轮，所以没有last10平均。最终变体选择仅使用FP32 dev-clean/other参考音素数加权PER，test不参与选择。

最终使用与原Libri100相同的6个manifest：Libri dev-clean/dev-other/test-clean/test-other及Buckeye人工标注dev/test。识别/无文本分割与给定音素native K2 FA独立列出。保持唯一区间端点、去句首句尾20ms、一对一10/20/30ms时间匹配、冻结IF-MDD R-value公式及FA duration/frame_count网格。Libri自动边界的来源模型身份未核实，不称为人工精度；Buckeye为相同原始人工标注19位子集。报告与加训前指标对照；F1/R保留全集，MAE用共同成功子集。

GPU门禁验证最长batch64有限loss/梯度、继承AdamW moments、保留冻结状态、按相同4-step schedule中断并恢复完整状态和同一W&B ID、四个完整epoch归档及8句真实native K2对齐。每组通过后才开始正式训练。

Miyabi三组为三个独立配置，各用一节点/一GPU，无DDP，不将三组当作一个模型同步。run_one.pbs为regular-g/24小时，22小时预算保存状态，剩余时间用于保存及评测。interactive.sh复用已经通过的原上游JSD验证并优先启动它；CE/standard在自己的生产节点上完成各自门禁；CE/standard独立提交，原上游JSD设置依赖接续。具体job ID和实时已保存状态见dispatch.json及各组训练日志。
