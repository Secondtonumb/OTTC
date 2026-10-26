# TIMIT Frame-wise CE：原四 seed sweep 的补充

参照 `../timit_seed_sweep_20261022/`，按用户要求仅新增3个seed：`frame_ce__seed42/43/44`，对齐原表三seed主表。
原表 `ce` 是 **OT 加权 CE**；本组是人工 PHN 时间标注提供的 hard frame CE，
使用更强的训练监督，比较时明确标注。未找到相同配置的已完成 Frame-wise CE，
不会用原 OTTC-CE checkpoint 冒充或继续训练本组。

匹配项：原始 facebook/wav2vec2-large-lv60 SSL 初始化、新随机 phone head、
42-ID 词表、native 10ms（CNN stride `[5,2,2,2,2,2,1]`）、前六层 CNN 冻结，
最后 CNN/Transformer/phone head 可训练。batch64，40 epochs/2920 steps，
LR2e-4、warmup500、weight decay0.005、clip1、BF16计算、FP32参数/保存/平均。
每个 seed 使用一个PBS单GPU节点的一张 GH200（当前实际路由队列small-g），world_size=1，无跨 seed 梯度通信。

数据直接复用原 manifests：train4620/dev400/test192，文件和词表哈希逐项核对。
训练标签取 `references.ottc_arpabet`，包括 sil/q/dx，序列逐项等于原训练 token IDs。
第 i 帧在原采样率16kHz上的感受野中心是 `199.5+160*i` samples；
取覆盖此中心的 `[start,end)` PHN 标签。保留原金标区间，不改时间戳或合并相邻区间。
遇到未知标签/重叠区间直接失败。原 manifests 中未覆盖的句首/句尾帧用-100忽略，
不填充猜测标签。CPU审核枚举了train28句839帧、dev4句131帧、test8句480帧；
完整明细见 `CPU_PASS.json`，各运行另存 `DATA_AUDIT.json`。

目标是每句话有效帧 CE 的均值，再对真实逻辑 batch 的句话取均值，
12句尾 batch 也按12归一化。无 OT、CTC、quantity、forward-sum 或其他辅助损失。
原 `lm_weight` 头保留为 frozen/unused，因本目标不使用 duration/mass head。
原始 CE 与本组 raw loss 含义不同，不横向比较其数值。

训练前依次验证 CPU gate 和 exact-workload GPU profile：原 SSL 全部 encoder tensor、
最长64/代表64/尾12真实音频、实际梯度、吞吐、显存、FP32保存及4句评分链路。
profile更新后的模型不用于训练，正式训练重新初始化。每个seed保存独立来源快照和配置。

每个完整 epoch 原子保存独立 `epoch_weights/epoch-NNN.pt`（含epoch/step/model/config/source）。
保留全部40份；`best_dev.pt`链接dev PER最优的epoch；每轮保存最新完整Trainer状态，支持同W&B ID续训。
完成后比较单best-dev与最后5/10轮FP32平均，平均记录成员/epochs/steps/SHA256。
先对全部候选评分dev并选择，再评分test。原表的主比较始终使用single best-dev，
平均结果单独列出，不能混入原表均值。

最终沿用原评分器：audio-only greedy PER/PanPhon PFER，text-free和supplied-phone FA各自的
P/R/F1/R-value（10/20/30ms），FA另报MAE。边界协议和scorer源哈希保留。
FA是历史CTC图在本组后验上的匹配诊断；frame CE没有blank监督，须注明解码器差异。
金标时间只用于训练标签及loss诊断/评分；音频推理不读取金标标签。

W&B只记录训练标量与配置；profile、dev/test预测、评分和最终报告均本地保存，
固定 `OTTC_EVAL_UPLOAD=0`，使用仓库 `wandb_policy.py` 约束评估上传。

运行：

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" -B -m reproduction.timit_frame_ce_20261026.validate
"$LIBRI_PYTHON" -B -m reproduction.timit_frame_ce_20261026.submit
```

`SUBMISSIONS.json`是提交凭证；profile/训练/benchmark分别以`PASS.json`、
`TRAIN_COMPLETE.json`和`BENCHMARK_COMPLETE.json`作为证据，不把queued/running视作完成。
最终 `REPORT.md` / `COMPARISON.json` / `RESULTS.csv` 会按完成seed更新，未完成保持缺失。

提交凭证：seed42=`3709328.opbs`，seed43=`3709329.opbs`，seed44=`3709330.opbs`。
当前核验状态见`STATUS.json`；原组实际用时及本组估计见`RUNTIME_REFERENCE.json`。
