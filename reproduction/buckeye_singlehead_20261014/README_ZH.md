# Buckeye 单头 OTTC CE / 原上游 JSD / standard JSD

沿用已有原始人工标注子集与固定划分：train 10,927句 / 15位，dev 1,524句 / 2位，test 1,840句 / 2位。源于完整压缩包中的完好录音，不是完整40位Buckeye；没有使用62.5ms量化镜像或历史处理版CTM作为参考。

保留自监督预训练 wav2vec2-large-lv60，随机初始化分类与帧质量头；CNN冻结，Transformer微调。三组 seed42、20ms、batch64、lr2e-4、AdamW weight_decay0.005、warmup500、40epoch（6840step），原上游regfrozen30epoch配方。CE是OTTC内部运输代价的CE版本，没有帧级CE。ao→aa映射与原TIMIT词表一致。

原上游JSD为0.5 KL(m||p)+0.5 KL(m||q)，沿用上游target epsilon；standard JSD为0.5 KL(p||m)+0.5 KL(q||m)，m=(p+q)/2。standard使用仓库现有jsd_standard分支，不重写公式。三种raw loss尺度不同，比较统一评测的PER、分割F1/R及K2 FA精度。

GPU验证发现原FP16配方在最长batch64上loss有限而梯度出现非有限值；三组统一使用BF16 autocast与FP32主权重。每组分别验证最长batch64、冻结阶段切换、backbone梯度及K2。该修改不改写运输代价或损失公式。

训练只接收音频和音素序列，三个split按说话人、句子、源录音隔离。dev做每epoch诊断，固定最后epoch40评测dev/test。报告分别列text-free识别/分割和K2给定音素FA；20ms一对一边界匹配、冻结IF-MDD R-value、全音素边界MAE。

`prepare.py`准备可审计manifest与train/dev音频缓存，不把时间戳放入训练缓存。CE/原上游JSD继续使用已有验证通过的`train.py`和`run.pbs`，不重新训练已完成模型。`standard_jsd.py`只扩展objective和来源哈希，保持共同训练源码不变。

`run_standard.pbs`先执行已有`check_js_cost.py`的SciPy/梯度检查，再分别运行standard训练smoke及8句dev的真实K2评测，通过后训练40epoch并评测dev/test。每次分配到时限前保存可恢复optimizer/scheduler/W&B状态；恢复入口相同。原源码/配置备份在`provenance_before_threeway`。

`summarize.py`验证三组配置、来源哈希、manifest、参考及逐句K2路径，分别列text-free和FA。MAE使用三组共同成功对齐句子，边界F1/R保留完整集合。尚有组未完成时返回成功并明确标注待完成，写`COMPARE_STATUS.json`；仅全部最终评测完成后写`COMPARE_COMPLETE.json`。最后完成的训练任务自动生成三组`REPORT_ZH.md`。

训练与评测均开启W&B。旧TIMIT权重、历史评分、Buckeye原始数据保持原样。配置和数据选择为当前默认；用户回复初始化/数据范围选项后，以其回复为准。
