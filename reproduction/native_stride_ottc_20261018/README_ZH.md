# 原生10/20ms OTTC：Wav2Vec2、S3T2** 与 WavLM Large

2026-10-18 用户依次要求测试 TIMIT S3T2**、训练10ms Wav2Vec2、去掉 upsample、增加10ms单音素 Wav2Vec2，以及 WavLM Large 的10/20ms × TIMIT/Buckeye/Libri100。最后明确每项均做 CE、原上游 JSD、standard JSD 三组。矩阵共24个实验位，见 `dispatch.json`；不同模型/目标/数据/步长各有独立 run。

## 模型和训练范围

- Wav2Vec2-Large-LV60：TIMIT原生10ms，单音素与S3T2**各三种代价。作为之前商定的小规模测试，从同一个 Libri960 原上游 JSD best-dev 初始化，追加10轮TIMIT训练，lr2e-5、warmup50。两种输出头对照具有相同 encoder 初始化，三种代价也共用同一初始化；这是迁移微调实验，不能冒称从自监督权重开始的40轮训练。
- WavLM Large：TIMIT、原始人工Buckeye partial19、Libri train-clean-100，各做原生10/20ms和三种代价。自官方 `microsoft/wavlm-large` 自监督权重开始，随机分类/质量头，40轮、lr2e-4、warmup500、前30轮学习质量分支，之后regfrozen。沿用已有验证的WavLM loader，真实encoder为WavLMModel。
- 10ms均直接把最后一个卷积的stride从2改为1，总stride320→160。20ms保持总stride320。未插值、未扩帧、未引入upsample层；encoder配置、实际卷积stride和输出长度同时验证。预训练卷积参数形状不变，CNN冻结、Transformer微调。
- 三种代价均调用未改动的 `ottc/ops_loss/ot_loss.py`。CE是OTTC内部运输代价，不是帧级CE；`jsd_upstream`与`jsd_standard`分别使用已有两条不同分支，不重写公式。
- S3T2**只复用冻结IF-MDD的子状态标签、状态展开与FSA图；普通音素三状态，sil一状态，保留q与重复音素occurrence。目标state-uniform，不包含额外FSA forward-sum训练项。原42类里的unk不进入119类子状态输出；blank保留。头从对应phone参数复制，bias减log(状态数)。

有效batch64、BF16/FP32主权重、AdamW decay0.005、seed42。微批量由数据和步长设定并接受最长输入检查；每个逻辑64句batch在Trainer内拆成微批量，按实际句数归一化并只更新一次优化器。最后不足64句的batch完整训练，不跨epoch累积或遗漏；每轮步数为73/171/446（TIMIT/Buckeye/Libri100）。

## 复用及数据

TIMIT为4620/400/192：历史训练集含924条SA，dev400与core-test192排除SA；Buckeye为原始人工19位子集，10927/1524/1840；Libri100为28538句训练，dev5567、test5558。训练只取音频和既有ottc_arpabet音素序列，时间戳仅用于独立评分。复用已有归一化音频缓存，不复制数据集。

实际核对旧缓存后，已有WavLM20ms/TIMIT JSD的4620条训练目标序列一致，但4452条音频长度与本批IF-MDD音频manifest不同，详见`DATA_MATCH_AUDIT.json`。旧权重和快照作为历史参考保留，不能充作本批匹配数据的控制。其旧kaldi-test240监控/固定final40选择也与本批dev400选择不同。本批使用独立名称开展相同音频manifest的新实验。历史WavLM20ms/TIMIT CE用batch128，同样保留为历史参考；本批CE为batch64。本次没有为补平均重启旧配方。

## Libri100启动条件

用户最后要求：只有TIMIT和Buckeye完整训练和评测结果正常、没有明显异常或实现错误后，才训练Libri100。当前6个Libri100实验全部保留，`--submit`只提交18个TIMIT/Buckeye实验。profile通过或短期loss下降不足以放行。

完成后检查数值/梯度与恢复、预测覆盖及插入删除、PER、无文本和给定音素分割P/R/F1/R与FA失败情况。人工审阅实际结果后，`LIBRI_GATE.json`必须记录检查结论和全部18个`BENCHMARK_COMPLETE.json`的SHA256；只有显式`--submit-libri-after-review`才能放行，缺少结果或任一检查未通过时拒绝提交。

## 验证、恢复和评测

`validate.py`先做原生帧长、三代价有限梯度、S3T2**重复音素FSA路径、连续与断点恢复逐参数一致性及独立epoch归档检查。每个生产作业之后还必须通过自己的全模型真实最长batch、代表batch和regfrozen阶段profile，记录loading/compute时间、显存、loss、梯度及GPU telemetry，才能正式训练。

每个完整epoch原子保存独立模型权重和epoch/step/config/source身份，全部保留。完整latest Trainer checkpoint包括优化器、调度器、RNG和采样恢复；另存best-dev。训练日志与W&B均记录raw loss、step、epoch、LR、PER、分割P/R/F1/R，恢复使用保存的run ID。网络失败保留offline日志。

完成后对best-dev、final、final-5和final-10平均做FP32 dev/test评测。平均成员记录路径、epoch、step及SHA256，先完成所有dev评分再做test，最终候选只按dev选。识别/无文本分割与给定音素K2 FA分开；S3T2**同时报告原greedy和音频phone-loop FSA。边界采用唯一端点、排除句首尾20ms、一对一10/20/30ms匹配及冻结IF-MDD R-value。子状态内部变化不算音素边界。Libri参考为既有自动对齐，模型身份未核实；不能当成人工边界精度。

## 执行

```bash
# GPU allocation内：实现/恢复验证
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" reproduction/native_stride_ottc_20261018/validate.py

# 登录节点：盘点或提交TIMIT/Buckeye，Libri100保留
python reproduction/native_stride_ottc_20261018/launch.py
python reproduction/native_stride_ottc_20261018/launch.py --submit

# 完整结果审阅通过并写入LIBRI_GATE.json后才可执行
python reproduction/native_stride_ottc_20261018/launch.py --submit-libri-after-review
```

单任务入口是`run_one.pbs`，通过`NATIVE_OTTC_RECIPE`指定矩阵中的完整名字。按用户2026-10-18的更正，每个作业一节点一GPU，regular-g/walltime=02:00:00。先profile再训练，二者合计预算100分钟，预留20分钟给保存和最终评测。预算耗尽保存后使用同一2小时入口接续；失败不会被当作完成。`TRAIN_COMPLETE.json`只表示训练完成，`BENCHMARK_COMPLETE.json`才表示本地最终评测完成，W&B上传状态另查日志和run记录。
