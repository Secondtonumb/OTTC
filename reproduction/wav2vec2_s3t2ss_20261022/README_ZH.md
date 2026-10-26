# Wav2Vec2 phonetic SPOTTC：S3T2**，CE / AG

2026-10-22：独立重做音素级实验，沿用当前 OTTC Wav2Vec2 backbone 配方。
两组均为纯 OT transport 目标，不含 forward-sum、CTC、逐帧 CE、平滑或其他辅助 loss。
CE 指 transport cross-entropy；AG 直接调用共享实现的 `divergence='ag'`，并验证与历史 `jsd=True` 完全一致。

## 共同条件

- 原始 facebook/wav2vec2-large-lv60 SSL 初始化，新建 substate 和 mass heads；不加载已监督 OTTC 权重。
- CNN stride `[5,2,2,2,2,2,1]`，原生 10ms；冻结前六层 CNN，训练最后一层、Transformer 和两个任务头。
- TIMIT train/dev/test 4620/400/192，人工 PHN 评分参考；同一音素库存、数据 hash、seed42。
- 40 epochs，logical batch64，AdamW，LR2e-4，warmup500，weight decay .005。
- FP32 参数、保存权重和 final5/final10 平均；训练 BF16 autocast，最终独立 FP32 评测。
- 每轮原子保存独立权重，并保存 best-dev 与完整 latest resume；保留全部40轮。

## 训练与解码拓扑的准确含义

训练复用原始子音素 OTTC：每个普通音素 occurrence 独立展开 `[state0,state1,state2]`，
`sil` 为单状态，不合并相邻重复音素；target mass 为 state_uniform，单调 OT 保持目标顺序。
这是 OT 的有序子状态目标，不是 FSA forward-sum；不会声称训练在图上边缘化了所有合法路径。

正式 audio-only recognition 使用冻结 IF-MDD `s3_t2_star_star` phone-loop 的 K2 Viterbi。
S3T2** 允许 state0 停留、中间 state1 跳过、state2 停留，完整音素必须从 state0 开始且由 state2 完成；
边界 blank、sil 单状态和相邻重复音素均按原图。greedy 仅为诊断。
forced alignment 使用同一供应者的 supplied-phone graph，单独报告，不与无文本结果混为一谈。

`topology_audit.py` 将短 acoustic 序列穷举，与独立状态机定义比较，并同时检查 phone-loop 和 supplied-phone graph；
验证重复音素展开、silence、合法 Viterbi 路径；用独立 dense interval-overlap OT oracle 验证 CE/AG loss 与 logits/mass 梯度。
`validate.py` 检查原始 SSL 全张量一致、训练/冻结范围、有限梯度、精确 resume 和逐轮保存。
每组 GPU profile 检查真实完整模型、microbatch16/32/64、最长输入、logical64、K2识别/FA和保存权重实际dtype，成功后才进入正式训练。

## 评估与追踪

每轮 dev：PER、边界 P/R/F1/R-value；训练 scalar curves 记录 W&B。
profile、最终 benchmark、预测、配置快照和完整 provenance 仅本地保存，`OTTC_EVAL_UPLOAD=0`。
最终比较 best-dev/final/final5/final10，先完成全部 dev 再评 test，按 dev 选配置。
`SOURCE_CONTRACT.json` 固定源码 hash；恢复时不匹配则停止，不修改历史 run 或 checkpoint。

入口为 `run_one.pbs`，每组单 GH200、两小时片段，正常限时保存后按同 W&B ID 自动续训。
提交记录见 `dispatch.json`；完成以 `TRAIN_COMPLETE.json` 和 `BENCHMARK_COMPLETE.json` 为准。
