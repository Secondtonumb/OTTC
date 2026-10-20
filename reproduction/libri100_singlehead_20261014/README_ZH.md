# Libri100 单头 OTTC 三组

沿用已完成 Buckeye 单头配方，训练数据改为完整现有 LibriSpeech train-clean-100：28,538句、100.59小时、251位说话人。官方 dev-clean/dev-other 共5,567句，test-clean/test-other共5,558句，说话人及音频互斥，无时长截断。

三组 CE / 原上游JSD / standard JSD 共用自监督预训练 wav2vec2-large-lv60、随机phone/mass单头，42类原OTTC词表、20ms、seed42、batch64、lr2e-4、AdamW weight decay0.005、warmup500、40epoch/17,840step、regfrozen30epoch、CNN冻结、Transformer微调、BF16 autocast及FP32主权重。只选择固定最后epoch40，test及Buckeye不参与选择。

CE为OTTC内部运输代价，没有帧级CE。原上游JSD直接复用原实现0.5 KL(m||p)+0.5 KL(m||q)，target epsilon不变；standard复用现有jsd_standard分支0.5 KL(p||m)+0.5 KL(q||m)。不修改核心模型/损失、不替换为IF-MDD标准JSD。SOURCE_REUSE.json记录Buckeye来源与适配，模型目录记录配置、源码哈希、快照及W&B身份。

已核查仓库此前Libri100 CE/JSD为四头模型，不能作为本单头同配方checkpoint。已有Buckeye/TIMIT模型、原数据及历史结果保持原样；本系列完成后复用，未完成从最新checkpoint恢复optimizer、scheduler、W&B原run ID。恢复时将模型内部零起始forward计数恢复为global_step-1，使冻结切换保持原40轮配方。

训练仅使用音频及既有canonical ottc_arpabet音素序列，含sil、映射后合并相邻重复。缓存只含input_values/labels/length，无参考时间戳。dev诊断每epoch保留原始loss、PER及分割P/R/F1/R；训练每50step记录loss、step、epoch及学习率。三种raw loss定义和尺度不同，不按raw数值直接排名。

Libri参考来自已有自动对齐（模型身份未核实），只映射ao→aa、zh→sh、去sil/q，不合并speech-phone时间区间。dev有一条仅sil的参考，保留该句；去sil PER按池化编辑距离计入其插入，给定speech-phone FA注明不可适用。原始manifest及音频不变。最终FP32独立评测分别报告dev_clean/dev_other/test_clean/test_other，保留全套逐句路径及指标，不能把这些自动参考指标称为人工标注分割精度。

另在与Buckeye-trained三组完全相同的原始人工标注子集dev1,524/test1,840句做零样本评测，独立列出。K2 FA给定音素、无参考时间戳、无sil目标；text-free指标另列。边界为唯一区间端点、去句首句尾20ms、一对一10/20/30ms匹配，R-value沿用冻结IF-MDD公式并记录源哈希。全集F1/R与共同成功子集MAE分开。单seed、最终checkpoint，未包含checkpoint平均。

Miyabi采用三个独立单节点/单GPU配置，无模型间DDP。实际生产入口run_one.pbs为regular-g/24h，预算23h时保存可恢复状态；run.pbs保留3节点MPI每rank一个配置的整套替代入口；同入口恢复。除每epoch外每200 optimizer step保存一次，保留5个可恢复checkpoint；最终epoch40仍单独保存。每个cost必须先通过最长batch64、有限梯度、冻结阶段切换及8句真实native K2验证。三个验证均成功才启动三组正式训练。W&B默认启用，沿用reproduction/wandb_config.json且尊重环境覆盖，网络不可用时保存offline日志。

interactive_start.sh先在可立即获得的单节点上依次验证三组，再优先启动原上游JSD；CE及standard各有独立batch，可在节点可用时启动；原上游JSD单独设置依赖接续，复用其checkpoint与相同W&B身份。所有进度写入dispatch.json及对应训练日志。每组完成后自动审计，最后完成的一组生成最终REPORT_ZH.md和COMPARE_COMPLETE.json。

`score_saved_fa.py`可在完整评测尚未结束时，从已保存的真实K2路径评分已完成的整个split，不重复GPU推理。它检查全量句子ID、原始参考、音素顺序、CTC路径及评测器来源；尚未写完的split明确标为pending。阶段性FA表为`FA_COMPLETED_SPLITS_ZH.md`，完整P/R/F1/R及同split当前完成模型的共同成功子集MAE为`FA_COMPLETED_SPLITS.json`。这不替代三组最终报告或其W&B上传。
