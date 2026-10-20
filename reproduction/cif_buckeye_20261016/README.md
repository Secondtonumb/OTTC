# Buckeye CIF 三组分段实验

用户要求：TIMIT 三组完成后，在 Buckeye 上继续三组对照训练，每组 40 epoch。

- 从同一 `wav2vec2-large-lv60` 预训练编码器独立初始化；不加载 TIMIT 训练权重。
- 原始人工标注 partial19，沿用既有 15/2/2 speaker-disjoint split；train/dev/test = 10927/1524/1840，约 7.334/1.110/0.870 小时。不是完整40位说话人 Buckeye。
- 使用 `reproduction/buckeye_singlehead_20261014/data` manifest；音素从人工 reference 直接构造训练目标，ao 已映射 aa，保留重复 occurrence，不添加人工句首/句尾 sil。与旧 Buckeye OTTC 的合并重复及人工 edge sil 目标有区别。
- 三组：CIF CE + quantity；再加 0.5 CTC；或再加 0.5 原上游 JSD OTTC。复用 TIMIT 的 CIF/decoder 和冻结的 archived OT loss，不替换 divergence。
- 40 epoch，batch64，每轮171更新，共6840更新；AdamW lr2e-4，warmup500，线性衰减，weight decay .005，seed42；BF16，冻结卷积前端，其余编码器可训练。三组独立 W&B run 与本地逐步日志；resume 保留同一 ID、optimizer、RNG、scheduler。
- 每轮 dev，按最大 dev text-free R-value@20ms 选择；结束后在该 checkpoint 上评测 dev/test。参考时间戳只用于评分，不输入模型。
- 冻结 IF-MDD directional greedy P/R/F1/R，10/20/30ms；预测为所有 CIF fire 除最后闭合事件，保留重复音素事件。分别报告 fractional 和 frame-grid 时钟。没有 forced alignment。
- 先等待三组 TIMIT 的40轮完成标记与完整最终评测，然后在 GPU 上跑13项 CIF 测试及三组各4步 batch64 最长音频 profile（最长26.586s），记录吞吐、加载/计算/优化耗时、显存和GPU遥测，评测dev/test各16条。只有 profile 成功且源文件校验通过才允许正式训练。
- `profile.pbs` 的 afterany 依赖只表示等待 TIMIT 作业结束；脚本内仍要求三组真正完成。`run40.pbs` 使用 afterok profile 依赖，profile失败则正式训练不启动。
- 新配方位于本目录，不改动正在运行的 TIMIT 配方或其 resume 合约。

提交记录见 `SUBMITTED.json`，数据审计见 `DATA_AUDIT.json`；profile结果见运行后生成的 `GPU_PROFILE_PASS.json`。
