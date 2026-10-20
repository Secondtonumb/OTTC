# TIMIT fused-phoneme：原仓库 Soft-DTW 复现

使用原仓库 `loss_type=softdtw`：CE 帧-音素代价、gamma=0.1、normalize=False，
保留原 CPU/Numba 动态规划与逐语音 loss 求和、batch 平均方式。
官方 Wav2Vec2-large-LV60 底座、seed 42、TIMIT train 4620 / kaldi_test 240
（含 SA）、官方 fused 映射 40 音素 / 42 输出、batch 64、40 epochs、
AdamW lr 5e-5 / wd .005、warmup 500 / linear scheduler、CNN 冻结。
学习率沿用已复现的 README CTC 基线；README 未列出独立的 TIMIT Soft-DTW
完整命令，因此它是原实现基线的复现，不是论文某个数值表格的逐项复现。
Soft-DTW 使用单个 linear 音素 head，不使用 OTTC 时间权重 head 或 regfrozen。
最终 epoch 40 / step 2920 用于测试，未按测试指标挑选 checkpoint。

必要兼容修复：将 Soft-DTW 接入已有 CTC 识别指标；克隆 labels 避免原地
改写标注；按每条实际帧数截断，避免 batch padding 掩码长度不一致。
Soft-DTW 动态规划和 CE 代价源码未改；已用独立 PyTorch DP 验证 loss 和梯度。

下表均为百分比。所有模型都只输入音频；分段为 argmax 类别变化（含 blank），
20ms 时间步，无拟合偏移，20ms 容忍的一对一边界匹配。
Overlap 为正确识别音素区间交集 / 全部参考时长。
这些分段指标与论文 starting-frame F1/IDR 不等价；结果来自单 seed，未做调参。
Buckeye 和 L2-ARCTIC 只用于补充测试。

| 模型 | 测试集 | raw PER ↓ | no-sil PER ↓ | boundary F1@20ms ↑ | R-value ↑ | interval overlap ↑ |
|---|---|---:|---:|---:|---:|---:|
| Soft-DTW CE gamma=0.1 | readme_kaldi_test_240 | 74.61 | 84.94 | 13.02 | 34.75 | 7.26 |
| Soft-DTW CE gamma=0.1 | timit_core_test_192 | 76.27 | 87.21 | 13.75 | 35.12 | 7.16 |
| Soft-DTW CE gamma=0.1 | buckeye_test | 91.19 | 89.05 | 12.23 | 34.32 | 1.31 |
| Soft-DTW CE gamma=0.1 | l2arctic_test | 82.31 | 88.54 | 13.44 | 34.97 | 3.54 |
| CTC | readme_kaldi_test_240 | 8.01 | 8.16 | 53.93 | 22.87 | 25.95 |
| CTC | timit_core_test_192 | 11.13 | 8.67 | 54.31 | 23.68 | 25.94 |
| CTC | buckeye_test | 52.16 | 26.98 | 47.01 | -6.85 | 17.44 |
| CTC | l2arctic_test | 40.85 | 23.38 | 47.55 | -1.51 | 18.55 |
| OTTC CE | readme_kaldi_test_240 | 8.61 | 8.94 | 62.35 | 67.63 | 71.28 |
| OTTC CE | timit_core_test_192 | 12.07 | 9.68 | 62.79 | 68.02 | 70.33 |
| OTTC CE | buckeye_test | 55.42 | 28.66 | 51.96 | 53.28 | 45.19 |
| OTTC CE | l2arctic_test | 41.97 | 24.31 | 61.38 | 63.77 | 54.17 |
| OTTC upstream jsd | readme_kaldi_test_240 | 8.61 | 8.87 | 65.29 | 70.18 | 73.68 |
| OTTC upstream jsd | timit_core_test_192 | 12.07 | 9.65 | 65.36 | 70.25 | 72.60 |
| OTTC upstream jsd | buckeye_test | 55.66 | 28.60 | 54.40 | 55.00 | 47.01 |
| OTTC upstream jsd | l2arctic_test | 41.67 | 23.69 | 64.57 | 66.39 | 56.48 |
| OTTC standard JS | readme_kaldi_test_240 | 15.53 | 17.20 | 61.53 | 67.43 | 70.14 |
| OTTC standard JS | timit_core_test_192 | 19.46 | 18.49 | 61.84 | 67.71 | 68.99 |
| OTTC standard JS | buckeye_test | 56.59 | 34.01 | 52.52 | 55.80 | 44.43 |
| OTTC standard JS | l2arctic_test | 43.11 | 29.82 | 62.20 | 66.13 | 52.85 |

- Soft-DTW CE gamma=0.1：训练 17.35 分钟，Trainer eval_wer 74.54%；checkpoint `/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_softdtw_fuse_phoneme/checkpoint-2920`。
- CTC：训练 12.94 分钟，Trainer eval_wer 8.11%；checkpoint `/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ctc_fuse_phoneme/checkpoint-2920`。
- OTTC CE：训练 13.87 分钟，Trainer eval_wer 8.65%；checkpoint `/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/checkpoint-2920`。
- OTTC upstream jsd：训练 13.85 分钟，Trainer eval_wer 8.61%；checkpoint `/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme_jsd/checkpoint-2920`。
- OTTC standard JS：训练 14.27 分钟，Trainer eval_wer 15.53%；checkpoint `/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme_jsd_standard/checkpoint-2920`。

## 结果解读与运行核对

该配置出现严重音素漏识别和长区间合并。README TIMIT 240 条参考共
9059 个音素，编辑对齐得到删除 5831、
替换 927、插入 1。
后验与分段对照图也显示部分长语音区间被分配给单一音素。
因此本次既没有达到 CTC/OTTC 的识别水平，也没有得到有用的分段效果。
这是原实现 gamma=0.1、lr=5e-5、seed 42 的结果；尚未搜索 gamma、学习率
或训练时长，不能据此判断 Soft-DTW 调参后的最佳表现。

TIMIT train 原始 PHN 核对为 4620 条、约 3.94 小时、平均 3.07 秒。
batch 64 对应每轮 73 步、40 轮 2920 步；GH200 上训练实测 17.35 分钟。
训练是官方预训练底座上的微调。最终 checkpoint 的 CNN 权重保持不变，
Transformer 第 1 层注意力和第 24 层前馈权重均已更新，核对记录见
`results/runexp_softdtw_fuse_phoneme/weight_update_check.json`。
音素输出分布见同目录 `phone_output_distribution.json`。
运行于 interactive 作业 3506088.opbs / mg0033，完成后已释放 GPU。

Trainer 的 eval_wer 是批量 padding 下的音素编辑率，与单音频 raw PER 分开保存。
完整逐语音预测、metrics 和后验图在 `results/runexp_softdtw_fuse_phoneme/`；统一 benchmark 图、收敛曲线和分段对照在 `results/softdtw_comparison/`。
