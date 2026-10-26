# Buckeye full40 单组 Frame-wise AG

> **已被新方案替换：** 用户改为四损失 × 三seed；原排队作业3715358已取消且未开始训练。新配置及进度见 [四任务方案](../buckeye_frame_sweep_20261027/README_ZH.md) 和 [汇总报告](../buckeye_frame_sweep_20261027/REPORT_ZH.md)。本文件保留原单组方案记录。

用户要求只增加一组，采用既有 frame-wise 散度对比中 dev PER 最低的 AG，固定 seed42。
选择依据是 TIMIT 三种子 single best-dev 的平均 dev PER：AG 8.1307%，
J-div 8.1722%，标准 JSD 8.4654%；不使用 test 指标选择损失或种子。

数据通过 `reproduction.buckeye_dataset.load_buckeye_rows` 及已验证 registry 解析，
沿用新版 full40 的 train18221/dev6417/test6839、24/8/8 speaker split、原42-ID词表及所有映射。
训练使用 `reference` 的原始时间区间，以 16kHz CNN 感受野中心 `199.5+160*i` 取逐帧标签。
不添加人工边缘 sil、不合并金标区间、不补全未标注间隙、不拆分复合标签、不改金标时间。
未覆盖帧使用 -100 忽略；未知标签或重叠区间立即报错。完整逐帧覆盖统计见 CPU_PASS/DATA_AUDIT。
这是使用 gold 时间标注的强监督补充，须与原 OT-only 模型分开解释。

沿用 TIMIT FrameDivergence 的原 AG helper、one-hot epsilon 和每句话有效帧均值，
再对逻辑 batch 的实际句数取均值。无 OT、CTC、quantity、smoothing 或其他辅助 loss。
原始 wav2vec2-large-lv60 SSL 初始化、随机 phone head、native10ms，前六层 CNN 冻结，
最后 CNN/Transformer/phone head 训练，mass head 冻结不用。
40 epochs，逻辑 batch64（尾 batch45），LR2e-4/warmup500/weight decay0.005/clip1。
单节点单GH200，world_size=1；参数、每轮保存和 final5/final10 平均全部 FP32，encoder BF16。

先通过 CPU 数据/损失/回调/精确 resume 验证，再在实际 GPU 上分别 profile 物理 batch64 和48。
两个独立 subprocess 测最长64、代表64、尾45、最长验证批、验证后反传、SSL初始化、梯度及保存恢复。
选择通过全部 gate 且代表64端到端耗时较短的 batch；均失败时不进入训练。
profile 权重不用于正式训练，训练重新初始化。

每个完整 epoch 原子保存独立 FP32 权重并保留全部40份，另存 best-dev 链接及最新完整 Trainer 状态。
续训复用同一 W&B run ID。训练只记录配置和训练标量至 W&B；所有验证、测试、预测、报告仅本地。
每轮用 Trainer dev PER 选择权重；避免重复跑一遍全dev音频和FA。
完成后独立 FP32 评分 best_dev/final5/final10，每个候选评估 Buckeye dev/test 和 TIMIT dev/test。
先 Buckeye dev 选择候选，test 和 TIMIT 从不用于选择。最终 PER/PFER、text-free 及 FA 的 P/R/F1/R-value 分开记录。
FA 使用已验证的无剪枝精确 CTC Viterbi；保留全部长尾样本，不按 gold 误差选样本。

运行：

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" -B -m reproduction.buckeye_frame_ag_20261027.validate
"$LIBRI_PYTHON" -B -m reproduction.buckeye_frame_ag_20261027.submit
```

作业请求8小时，训练6小时预算到期会保存并以75退出，避免将暂停称为完成；可用同一入口和run ID续训。
进度看 `jobs/<jobid>/worker.log`、`models/frame_ag__seed42/history.jsonl`；
GPU gate、训练和全部评估分别以 `PASS.json`、`TRAIN_COMPLETE.json`、`BENCHMARK_COMPLETE.json` 为证。
最终结果见 `REPORT_ZH.md` / `RESULTS.json`，单 seed 不填三种子误差条。
