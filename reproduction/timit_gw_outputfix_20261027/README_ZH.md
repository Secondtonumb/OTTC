# TIMIT GW 完整 40 epoch（2026-10-27）

用户要求完整训练，替代前一轮仅 5 epoch pilot。

- 方案 2：原 `reproduction/timit_gw_20261026/models/monotone_w0.1__seed43/checkpoint-365` 续训至 epoch40；保留原源码/hash/resume/optimizer/scheduler/W&B ID。
- 方案 3：本目录 `asot_w0.1__seed43`，原 SSL 初始化，完整 40 epoch。
- solver 控制组：本目录 `asot_w0__seed43`，同上。

修复对象是 CausalLMOutput 的 mapping/attribute 不同步：原 ASOT 分支从 audio-only 输出添加 loss 属性，未添加 loss 字典项，Accelerate 重建后 loss 变成 dict。
修复模块 `ottc/models/sequence/wav2vec2_ottc_gw_outputfix.py` 重新构造 CausalLMOutput；不改张量、数学公式或梯度。
原失败记录、原模块和完成的单调 pilot 保留不动。失败 ASOT 无 optimizer checkpoint，修复版从相同 SSL/seed43 开始，不声称续训。

15 项 CPU 检查通过，包含两个 ASOT 权重的真实 BF16 QuietTrainer optimizer steps；新 GPU profile 也显式经过 Accelerate 的输出转换。
训练沿用 batch64、native10ms、单头 AG、warmup500、原 40 epoch scheduler。所有 parameters/epoch weights/平均累积保持 FP32。
每 epoch 保存权重，另保留完整 latest resume 和 best-dev；最后评估 best/final/final5/final10，dev 选模型，test 单独报告。
训练标量 W&B；评估/预测/表格 local-only。

原 profile representative batch64：单调 1.19 s，ASOT 2.71/2.78 s。给单调续训 2h、ASOT 各 4h（训练预算12600秒，余量用于最终评估）。排队时间无法保证。

`resume_monotone.pbs` 运行原入口、不重复 pilot；`run.pbs` 先新 profile，再完整训练，不带 --pilot。
`report.py` 汇总原单调模型、新 ASOT 模型及历史同 seed AG。
`FULL_COMPARISON.json` 分别保留 best-dev 对比和 dev-selected averages；没有完整指标文件时只标 incomplete。
诊断：短 A-B-A、插入、短 gold 音素删除率、text-free PER/P/R/F1/R-value，与 forced alignment 分开。

5 epoch 初步结果（同400句dev）：AG PER64.0302、I812、D1443、短ABA154、短phone删除417/1490；单调GW PER63.0924、I707、D1523、短ABA150、短phone删除440/1490。
这是 warmup 内的混合结果，不能据此确认完整训练有效。ASOT 原失败不是模型效果结果。
