# OTTC 音素模型、权重与复现审计（2026-10-07）

## 权重查找结论

检查的官方源码为 `idiap/OTTC`，commit `0d97e7e`。仓库没有权重、词表、训练结果或本地 `path_config.py`。

公开检索中暂未找到能确认属于原论文的 TIMIT OTTC checkpoint：

- [论文 v3](https://arxiv.org/html/2502.01588) 只提供 GitHub 链接，没有 Hugging Face/checkpoint 链接，也没有 GPU 型号或训练耗时。
- [官方 GitHub releases](https://api.github.com/repos/idiap/OTTC/releases) 与 issues 查询均为空；主分支只有模型代码。
- Hugging Face 模型搜索 `ottc` 返回 [Haopeng/l2_arctic_OTTC](https://huggingface.co/Haopeng/l2_arctic_OTTC) 和 [Haopeng/iqra_OTTC](https://huggingface.co/Haopeng/iqra_OTTC)。搜索 `temporal transport`、`Optimal Temporal`、`kaloga` 没有找到相应模型；Idiap 公开模型列表中未找到 OTTC。此结论是当前公开检索结果，不等于证明任何其他命名或私有仓库都不存在。

`Haopeng/l2_arctic_OTTC` revision 为 `bd8c377a68631d47ef5ff35011b452a2922fa951`，采用 SpeechBrain 的 `model.ckpt`、`perceived_ssl.ckpt`、`tokenizer.ckpt`。发布的 inference.yaml 有 IF-MDD 本地路径、WavLM-large、384 维 MLP，而且写着 `ctc_loss_type: crottc` 和 IQRA 数据路径。因此不能凭仓库名称认定为官方 Wav2Vec2/TIMIT checkpoint，也不能直接相信这个 YAML 的加载配置。

本地已有可进一步评测的自有 OTTC 变体：

| 本地目录（相对 IF-MDD） | 模型配置 | 已记录 checkpoint metadata |
|---|---|---|
| `pretrained_models/L2arctic_acou_model/l2arctic_OTTC` | 有约 1.26 GB SSL 权重、2.24 MB 下游权重；需核对实际加载结构 | epoch 140，PER 17.5440，mpd_f1 0.71088 |
| `pretrained_models/librispeech_small_acou_model/ottc` | WavLM-large + 384 维 MLP + 音素/权重 head；有完整 SSL/downstream 权重 | epoch 97，PER 8.08778 |

这些 metadata 是既有文件的记录，未在本轮重新计算，也不是 TIMIT 测试结果。

## 原论文与代码的复现范围

论文使用约 300M 参数的 Wav2Vec2-large，底座来自 60K 小时英语自监督预训练。复现所需的是在 TIMIT 上微调，无需重做底座预训练。本地已存在底座：`/work/gm64/m64000/reborn-eval-20261007/models/upstream`。

TIMIT 微调设置：AdamW，学习率 2e-4，500-step warmup，linear decay，40 epochs；前 30 epochs 学习识别和对齐，最后 10 epochs 固定对齐权重，greedy decoding。官方 README 指定 batch 64、weight decay 0.005、冻结 CNN。

已核查本地原始 TIMIT：

| split/协议 | 语音数 | 音频时长 |
|---|---:|---:|
| 官方 loader 使用的完整 train（包括 SA） | 4,620 | 3.9424 h |
| 完整 test | 1,680 | 1.4407 h |
| 官方 loader 的 24-speaker kaldi_test（包括 SA） | 240 | — |
| 现有 benchmark 的 train（排除 SA） | 3,696 | 3.1451 h |
| 现有 benchmark 的 dev / core test | 400 / 192 | 0.3425 / 0.1619 h |

`timit_phoneme` 保留原始 61 类；`timit_fused_phoneme` 应用 allophone/closure 映射并合并相邻相同音素。当前官方映射保留 `q`，实测 40 个音素类型，加 pad/unk 后 42 个类别。它与本地 39-phone 协议不相同。

正式复现前需要解决的具体问题：

1. 论文写 logits 使用 linear head、OT 权重使用两层 linear + GeLU；代码 `final_layers=linear` 是两边都 linear，`mlp` 是两边都 MLP + ReLU。没有完全匹配论文的现成选项。
2. `ottc/utils/eval_metrics.py:188` 将整个额外输出 tuple 赋给 `weights_logits`，应核查到实际权重数组；`forward()` 在 `labels=None` 时仍先操作 labels，无参考转录推理入口需要补全。
3. TIMIT loader 的最终 map 移除原始标注列，包括 `.phn` 边界。分段评测需要另行保留 utterance ID 和原始 timestamp。
4. 当前已有环境的 Datasets 4.x 不支持该代码依赖的 `load_metric` 与旧式 dataset script API，建议按官方 Datasets 2.18 / Transformers 4.39.3 建独立兼容环境。
5. 固定第 30 epoch 对齐的 step 数、恢复 checkpoint 的 step、选 final/best checkpoint，都应显式记录。论文报告 final epoch，不能默认换成 dev-best 来宣称复现。

## 已有 benchmark 的接入

本轮记录了精确 manifest 路径、SHA256、schema 和 utterance 数，见 `benchmarks.json`：TIMIT dev/test 400/192，Buckeye local test 1,840，L2-ARCTIC local test 900。

建议分别报告两项能力：

- 识别：输入只有音频，greedy 音素输出，使用同一音素映射和静音策略计算 PER；同时保留原始输出与编辑计数。L2-ARCTIC 的 perceived reference 使用 `perceived_train_target`，其时间戳配对为 `target_starts/target_ends`。保留 `err` 的评分契约须明确。
- 分段：先报告音频独立预测的音素变化边界，P/R/F1@10/20/30ms、R-value、over-segmentation；另报标签匹配的区间 overlap/TSE。参考转录参与的 forced alignment 单独报告。

可复用的现有产物：

- `/work/gm64/m64000/reborn-eval-20261007/RESULTS.md`：同样三个测试集，音素 PER + 无转录 raw boundary 指标，有完整推理结果与 provenance。
- `IF-MDD/fa_research/scripts/evaluate_fa_boundary_prf_rvalue.py`：边界和区间评分。
- `IF-MDD/fa_research/scripts/evaluate_json_alignment_fa.py`：既有 OTTC/CTC 变体的参考转录条件 FA。它默认 Buckeye 全集 7,946 条，与上述 1,840 条 local test 不同。
- `IF-MDD/fa_research/results/zero_shot_per_mdd_20260626/metrics/summary.json`、`IF-MDD/timit_pfer_metrics_20260625_closure_variants/SUMMARY.md`：跨模型识别及 closure 协议对照。

论文 TIMIT alignment 表报告 OTTC F1=89.27、IDR=76.72、peaky=0.76，但 F1 定义为 token starting-frame accuracy，不能直接与上述无转录 boundary F1@20ms 相比。论文表 2 将 TIMIT 8.76 标为 WER；正文又说明使用音素 token，确认作者的实际评分定义前，不把 8.76 擅自标成标准 39-phone PER。

## 训练耗时计量

单 GPU GH200 120GB 的计时脚本、PBS wrapper 和输出分别为 `time_timit.py`、`time_timit.pbs`、`timing.json`。使用原版模型/loss，batch 64、FP16、源码启用的 gradient checkpointing，4 个按时长排序的真实训练 batch，分别测前 30 epochs 和后 10 epochs 的训练步骤。每个 shape 预热一次，再测两次 optimizer step。计时权重不保留，正式 40-epoch 训练未启动。

最终采用已完成的 job `3499463.opbs`，GH200 节点 mg0033，Exit_status=0，任务 walltime 29 秒。旧 LV60 checkpoint 的 weight_norm 名称与 Torch 2.6 有兼容差异，计时加载器仅在内存中转换 key。底座 loading_info 无 missing/mismatched/error；unexpected keys 只有不用于 Wav2Vec2Model 的原始自监督 quantizer/project heads。两阶段均确认 encoder 第 0/23 层和识别 head 的梯度存在、非零且有限。

| 计量项 | 结果 |
|---|---:|
| 前 30 epochs 平均训练 step | 0.1931 s |
| 后 10 epochs 平均训练 step | 0.2568 s |
| 40 epochs，73 batches/epoch，纯训练外推 | 10.17 min |
| 模型 + optimizer checkpoint 字节数 | 5.01 GB |
| 一次共享文件系统 checkpoint 写入 | 2.10 s |
| 每 epoch 保存一次的 40 次写入外推 | 1.40 min |
| 抽样最长 4.928 秒 batch 的峰值 GPU allocation | 15.56 GB |

建议单个 TIMIT OTTC 40-epoch 完整运行暂按 **20–40 分钟**估算，初次正式任务申请 **1 小时、单节点单 GPU**。这是带余量的规划估计：本轮未测完整 HF Trainer、数据 map、所有最长 batch 或逐 epoch 分段评估，不能把 20–40 分钟说成完成长训后的实测；大量额外分段/跨数据集评估需单独计时。CTC+OTTC 两条独立实验可分别运行，规划 1–2 GPU 小时有余量。环境兼容和评估入口修复所需的人工作业时间不计入 GPU 训练时间。

`timing.json` 的 `trainable_parameters` 是源码 `requires_grad=True` 的总数，包含备用的 `wav2vec2_freeze` 拷贝；源码对 module 属性赋值并不逐个冻结 Parameter。实际被冻结支路在 `torch.no_grad()` 中计算。因此它不是活跃训练参数量，原始底座依然约 300M。

外推公式：`ceil(N_train/64) * (30*t_joint + 10*t_frozen)`；加上每 epoch 的评估、checkpoint 写入、数据读取与初始化才能成为完整运行耗时。排队时间另算。这个是短计时外推，不是完成 40 epochs 后的实测。

计时调度采用 [miyabi-dispatch skill](/home/m64000/.codex/skills/miyabi-dispatch/SKILL.md) 的单节点验证方式（debug-g）。正式 40-epoch 运行建议 regular-g，沿用单节点配置并将 walltime 改为 01:00:00；本轮没有提交正式训练。
