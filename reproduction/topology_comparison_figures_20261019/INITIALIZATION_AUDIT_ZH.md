# 初始化口径核查：10ms 与历史20ms

本次用户质疑“不是 from scratch 训练的吗？”后，重新核对实际加载代码和保存配置。

- 当前10ms Wav2Vec2 monophone upstream-JSD：不是从原始SSL checkpoint重新训练OTTC。实际加载已经完成LibriSpeech监督OTTC训练的Libri960 upstream-JSD best-dev模型，再迁移TIMIT训练10轮；保留原音素输出头和frame-mass分支，后者参数被冻结。
- 历史20ms monophone upstream-JSD：从原始facebook/wav2vec2-large-lv60自监督预训练编码器出发，新建OTTC任务头，在TIMIT训练40轮。不是全模型随机初始化，也不是继承Libri监督OTTC模型。
- 当前六组Wav2Vec2（mono/S3，三种代价）均采用同一个Libri960 upstream-JSD监督parent；S3输出头由parent音素头复制扩展。因此CE/standard-JSD也不是各自从SSL重新训练的独立对照。
- WavLM分支的初始化路径不同，不应套用上述Wav2Vec2监督parent描述。

证据：

1. `../native_stride_ottc_20261018/models/wav2vec2__timit__10ms__mono__jsd_upstream/run_config.json`：initialization明确为Libri960 upstream-JSD best-dev supervised checkpoint，parent SHA256为90b4bde0eafa78494af617a17361ec0146a1a5e9e73daea1cfa0f860501d48f7。
2. `../native_stride_ottc_20261018/common.py:138-145`：`wvForOTTC.from_pretrained(parent['checkpoint'])`加载整个已训练OTTC模型；S3 head复制扩展及mass分支冻结见同函数后半部。
3. `../README.md`的原始单头复现与CE/JS比较章节、`../train_js.pbs`、`../../run_exp.py:132`和`../../ottc/models/sequence/wav2vec2_ottc.py:158`：历史20ms复用原始SSL编码器，新建OTTC任务训练。

此前图表数值仍是这些实际配方的有效测量，但不能解读为同初始化、同训练配方下的10ms/20ms消融。当前10ms的PER14.75%、无文本F1@20ms58.31%应标为“Libri960监督OTTC迁移TIMIT”，不能称为“从原始SSL重新训练OTTC”的结果。

本次仅核查并澄清初始化；未启动任何新训练，未修改旧权重、源码快照或测量结果。
