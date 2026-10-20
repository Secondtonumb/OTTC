# TIMIT 20 ms monophone CROTTC

用户已明确三组为 upstream JSD、CE、standard JSD，并授权接入 IF-MDD 的 speech augmentation。三组使用独立目录和 W&B run；当前实际阶段见 `STATUS.json`、`dispatch.json` 和对应作业日志。CPU 通过只说明实现门控通过，GPU profile/正式训练/最终结果分别查各自标记。

## 复用边界

复用 IF-MDD 配方的 SpeechBrain 1.0.3 `SpeedPerturb`、`DropFreq`、`DropChunk`、`Augmenter`，参数和上游源码 hash 见 `vendor/provenance.json`。先对完整逻辑 batch 共享一次 `[95,100,105]` 语速扰动，再独立采样两次增强。沿用 Augmenter 默认：顺序保留，每路随机取前 1–2 项，故 DropFreq 每路启用、DropChunk 随机启用。没有额外添加 YAML 中未接入 augmentation 列表的 SpectrogramDrop 或 timewarp。编码器原有内部 dropout/SpecAugment 配置保持原始 SSL 设置。

两路增强使用相同语速和长度，按 `[view1(batch), view2(batch)]` 拼接送入同一编码器。验证/测试只用原始单视图音频，关掉外部增强和 CR；clean eval raw loss 是单视图 OTTC，不能当作训练总 loss。

冻结保存 IF-MDD `CTCLossManager._compute_cr_loss` 方法的源片段，保留 teacher detach、双向 KL 及历史分母 `B*T*C-padding_frame_count`，不悄悄改成有效元素数归一化。默认 `cr_loss_weight=0.1`。三组 transport 均调用 OTTC 仓库原有 `ottc/ops_loss/ot_loss.py`：`jsd=True` / `False` / `'standard'`，没有导入 IF-MDD 的 transport 公式。

完整逻辑 batch 目标为：

```
raw_total = 0.5 * (OTTC_view1_sum + OTTC_view2_sum) + 0.1 * legacy_CR_mean
optimizer_loss = raw_total / number_of_source_utterances
```

这是历史 IF-MDD 两项相对比例，加上现有 OTTC Trainer 的按原始句数整体归一化；不是完全相同的 SpeechBrain 优化器配方。历史 CR 的分母随类别数/padding 改变，其相对 OTTC batch sum 的权重也较小。本次保留该语义以便追溯，每次实际 optimizer update 都记录原始 CR、加权 CR、OTTC、total 和归一化值。

显存不足时仅拆分编码器计算：先为完整 batch 生成增强、统一 padded 时间网格，再成对切 microbatch；每块 CR 按完整逻辑 batch 分母加权。不会为各 microbatch 重新独立挑语速，也不会对每块 CR 平均后再次平均。确定性小模型验证完整 batch 与 microbatch 的 loss/梯度一致性。

## 模型与数据

- 原始 `facebook/wav2vec2-large-lv60` SSL 权重初始化；全编码器 state_dict 逐张量核对，任务 phone/mass 头新随机初始化。没有加载已监督训练的 OTTC checkpoint。
- 原生卷积 stride `[5,2,2,2,2,2,2]`，16 kHz 对应 20 ms；没有 upsample；单音素 fused40，42 logits 含 blank/unk，phone-uniform transport mass。
- CNN 全部冻结，Transformer 与两个任务头全程训练。采用 IF-MDD 式 `regular`，不在第30轮切换到 regfrozen。它与原始20ms OTTC的 regfrozen 配方存在明确差别，历史无CR权重只能作为历史参考，不能称为严格的纯CR消融。
- TIMIT 4620/400/192，复用 `reproduction/multihead_timit/data` 清单和已有音频，不复制数据。训练只用音频与既有音素序列，参考时间只用于独立评分。
- seed42，40 epochs，logical batch64，AdamW lr2e-4、warmup500、weight_decay0.005；BF16编码器，FP32 transport/CR计算，最终独立 FP32 评测。

原有完成的 CE 和所有活跃配方/权重不修改、不重训。CROTTC-CE 是用户本次指定的新目标，不是为补平均而重跑旧 CE。复用核对见 `REUSE_AUDIT.json`。

## 门控、资源及恢复

CPU 门控验证真实增强双视图、相同时间轴、padding、双向 teacher detach、三种代价的非零有限梯度、CNN 冻结、microbatch 缩放、独立 epoch 权重以及包含增强 RNG 的逐参数精确恢复；另核对完整原始 SSL 权重。

每组一张 GH200、32 CPU，regular-g 两小时。在生产前先通过 CUDA 小模型门控和该组完整真实模型 profile，测代表 batch64、最长64句和最长语速、microbatch4/8/16/32/64 的前向/反向/optimizer、加载/传输/增强时间、显存、吞吐、GPU telemetry、真实 dev8 K2 FA，以及权重写盘。按安全吞吐选 microbatch，profile 模型随后丢弃，生产重新初始化。没有 GPU profile PASS 不会开始生产。

profile＋训练目标预算100分钟，剩余20分钟供保存及评测。未完成时保存完整 Trainer/optimizer/scheduler/RNG 状态，使用相同 recipe 和 W&B ID 接续两小时作业。失败不会自动冒充完成；正常预算暂停才会提交后续作业。`TRAIN_COMPLETE.json` 和 `BENCHMARK_COMPLETE.json` 含义不同。

每个完成 epoch 原子保存独立模型权重、epoch、optimizer step、config/SSL/source 身份，保留全部40轮；另存 best-dev 和完整最新状态。最终比较 best-dev、final、final5/final10 平均，记录平均成员的路径/epoch/step/hash。先评分全部 dev 候选，再评分 test，选择只依赖 dev。

## 指标与 W&B

dev400 与 core-test192 分开。音频识别 PER / 无文本分割与给定音素 K2 forced alignment 分开；报告 P/R/F1/R-value@10/20/30ms 及 FA MAE。边界唯一端点、排除首尾20ms、一对一匹配；R-value 复用冻结 IF-MDD 公式。无文本按原生固定20ms时钟，FA按 duration/frame_count、blank归前音素。不给识别路径参考文本或时间戳。这里不与身份未核实的历史MFA CTM作topline结论。

默认 W&B 项目/entity 来自 `reproduction/wandb_config.json`，尊重环境覆盖。profile/training 各独立 run；resume 复用 ID。记录 raw loss 分量、epoch/step/LR、实验配置和完整本地日志；联网失败保留 offline 记录。最终 dev/test 数值汇总上传 W&B artifact，不上传音频或源码包。

```bash
source reproduction/libri960_continue5_20261017/environment.sh
"$LIBRI_PYTHON" reproduction/crottc_timit_20ms_20261019/validate.py --device cpu --full-ssl
"$LIBRI_PYTHON" reproduction/crottc_timit_20ms_20261019/launch.py --submit
"$LIBRI_PYTHON" reproduction/crottc_timit_20ms_20261019/status.py
```
