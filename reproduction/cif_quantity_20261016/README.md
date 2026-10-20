# CIF：Buckeye / Libri100 三组，全部保留 Quantity

用户明确澄清：三组都保留Quantity。每种训练语料各三组、每组40 epoch，共六组：

|内部variant名|训练目标|Quantity权重|
|---|---|---:|
|ce|CE + Quantity|1|
|ctc_0p5|CE + Quantity + 0.5 CTC|1|
|ottc_jsd_0p5|CE + Quantity + 0.5 原upstream-JSD OTTC|1|

- 主目标为 `(CE_sum + Quantity_L1_sum + 0.5 CTC_sum)/Nphones + 0.5 OTTC_native_sum/Nutterances`，两种辅助分支互斥，不存在的项为0。Quantity使用训练缩放前的原始alpha总和与真实音素数量的L1差。训练时CIF按目标长度缩放；纯音频推理不输入文本或真实数量。
- 所有正式入口显式传 `--quantity-weight 1`；训练器拒绝其他Quantity权重。启动前校验六组profile的配置、源码及数据，并核对实际loss中Quantity的贡献。
- 从相同预训练Wav2Vec2-large-LV60、seed42独立初始化。CE label smoothing .1；batch64，AdamW lr2e-4，weight decay.005，warmup500，40轮线性衰减，BF16/FP32，CNN冻结、Transformer全程训练。没有继承Q0模型/optimizer。
- OTTC沿用冻结原上游实现 `archived_ot_loss.py` 的jsd=True；独立phone logits头与独立Linear(H,1)+时间softmax质量头。CIF的sigmoid积分alpha保持独立。与旧原生OTTC的regfrozen及decoder不同，不把差异仅归因于loss。
- Buckeye：原始人工partial19，train10927/dev1524/test1840，15/2/2说话人互斥；训练7.334h，最长26.586s。保留重复人工phone occurrence，不添加人工edge sil。40轮共6840更新。
- Libri100：现有完整train-clean-100 manifest28538条、100.586h；dev-clean+other5567，test-clean+other5558，官方说话人互斥，训练最长24.525s，held-out最长35.155s，无时长截断；40轮共17840更新。其边界来自已有自动对齐，模型身份未核实，不能称人工分割真值。
- canonical ottc_arpabet目标及其源参考一致，Libri保留源sil；零时长边缘sil（337/31/43）保留在文本目标中，分段评分排除句首/末端点。原始数据未改动。
- 每epoch完整dev，按最佳dev text-free R-value@20ms选checkpoint；40轮后评测完整dev/test，并单列Libri clean/other以及同一原始人工Buckeye dev/test，区分域内/跨域，测试集不参与选择。
- 音频独立分段：所有CIF fire除最后闭合事件，重复保留；duration/T时钟，fractional/right-cell分开；冻结IF-MDD directional greedy P/R/F1/R，10/20/30ms。评测batch8，CPU验证与batch1的tokens/边界/CE一致；没有forced alignment。
- 每个run独立W&B ID，恢复保留optimizer/scheduler/RNG/ID；记录loss各分量、epoch/step/LR、PER及P/R/F1/R。网络失败留offline日志。完成后上传最终benchmark summaries。
- 精确配方先通过15项模型/评测测试及六组各4步最长batch64 profile、dev/test各16条、最长held-out batch8；记录计算/加载/优化时间、吞吐、峰值显存和GPU遥测。数据header审计复用同一未改变manifest的现有审计，重新核对hash。
- `cif_noquantity_20261016` 是助手对“Quantity先不用管”的误解所产生的Q0实验，已撤销全部作业，保留日志/checkpoint但排除正式比较。本目录是纠正后用户要求的Q1实验，与TIMIT已完成Q1实验一致地保留Quantity。

`REQUEST.json`记录纠正后的要求；`DATA_AUDIT.json`和`SHORTEST_BATCH_SAFETY.json`记录数据/批次安全性；`GPU_PROFILE_PASS.json`与`VALIDATION_PASS.json`是正式启动门槛。提交和实际进度分别见`SUBMITTED.json`、各run的history/checkpoint/完成标记。runtime时间预算只负责分段接续，不改变40轮目标或LR schedule。
