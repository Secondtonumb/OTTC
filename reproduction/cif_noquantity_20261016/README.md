> 已撤销：用户澄清三组都应保留Quantity。本目录Q0由助手误解产生，作业已停止，日志/checkpoint仅保留审计，不属于用户要求的正式对照。正确配方见 `../cif_quantity_20261016/README.md`。

# CIF 无 Quantity：Buckeye 与 Libri100 三组对照

用户最新决定：各做 CE、CE+0.5 CTC、CE+0.5 原 upstream-JSD OTTC，Quantity loss 权重全部为0。共六个独立训练实验，各40 epoch；两种辅助loss互斥。

- CIF仍在训练时按真实目标长度缩放积分质量；纯音频推理不提供真实音素或长度。Quantity原始误差只作诊断，未加入训练目标。CIF框架的无Quantity版本不称为原论文完整复现。
- OTTC复用冻结原上游实现 `reproduction/zapformer_s3t2_native_20261012/archived_ot_loss.py`，独立 `Linear(H,1)` + masked temporal softmax计算OTTC alpha，另有独立phone logits头；CIF sigmoid alpha仍独立。此版本没有把OTTC alpha与CIF alpha共享。
- 三组总目标分别为 `CE_sum/Nphones`、`(CE_sum+0.5 CTC_sum)/Nphones`、`CE_sum/Nphones+0.5 native_OTTC_sum/Nutterances`。CE label smoothing=.1。不同raw总loss不直接比较大小。
- 同一预训练Wav2Vec2-large-LV60，seed42，batch64，AdamW lr2e-4、weight decay.005、warmup500、线性衰减到40轮，BF16/FP32混合；冻结CNN，Transformer全程训练，无regfrozen。每个run保存optimizer/RNG/scheduler和W&B ID以恢复。
- Buckeye：复用原始人工标注partial19，train10927/dev1524/test1840；15/2/2说话人互斥，约7.334h训练，最长26.586s。保留重复phone occurrence，不使用旧配方的人工edge silence目标。
- Libri100：复用完整train-clean-100 28538条、100.586h；dev-clean+other5567，test-clean+other5558，官方说话人互斥；训练最长24.525s，dev/test最长35.155/34.955s，无音频截断。canonical ottc_arpabet IDs及其源参考严格一致，保留源sil。其时间标注来自已有自动对齐，模型身份未核实，不能称人工边界真值。
- Libri源含零时长句首/末sil（train337/dev31/test43）：保留原canonical序列；分段评分排除时间0与末端。这些sil不算额外内部边界。数据、原始标注和旧checkpoint均未改动。
- 每epoch完整dev评测，按最大dev text-free R-value@20ms选checkpoint。全部40轮后评测完整dev/test，并另列Libri clean/other和同一原始人工Buckeye dev/test，支持训练域内与跨域比较；跨域集合不参与选点。
- 分段来自CIF fire位置（除最后闭合事件），保留重复预测，duration/T时钟，fractional/right-cell grid分别报告；冻结IF-MDD directional greedy P/R/F1/R，10/20/30ms，哈希写入指标。它是无文本分段，不是forced alignment。
- 评测按长度分桶、batch8；CPU测试验证batch1与batch4的预测/边界/CE一致。正式训练前在交互GPU对六个精确配方分别做4步最长batch64、dev/test各16条的profile，并检查最长held-out batch8。记录加载/计算/优化耗时、吞吐、峰值显存及GPU遥测。
- W&B默认开启，项目/entity继承仓库配置，六组独立run；记录raw loss及CE/Quantity/CTC/OTTC分量、step、epoch、LR、PER与P/R/F1/R。完成后上传原始最终评测summary。联网不可用自动保留offline记录。
- 旧Buckeye带Quantity作业3567088/3567089/3567090在Q状态撤销，未启动正式训练。原TIMIT三组已完成结果及旧profile保留。本目录是用户新要求的独立Q0实验，不能与旧Q1结果混报。

`REQUEST.json`记录决定，`DATA_AUDIT.json`记录音频/划分审计；`GPU_PROFILE_PASS.json`是正式启动门槛；作业提交与实际训练进度另存`SUBMITTED.json`和各run的`history.jsonl`。运行时可从相同入口恢复，未出现`TRAIN_COMPLETE.json`及完整最终评测前不宣称完成。
