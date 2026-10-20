# TIMIT fused-phoneme：CE 与 JS 代价对比

原 README 复现使用 CE（`bins_metric=crossentropy`）。本次追加两个独立实验，
均从相同官方 Wav2Vec2-large-LV60 底座、seed 42 开始，未从 CE 模型续训。
除音素 transport cost 外，TIMIT 数据、40 音素映射、regfrozen 2165 steps、
batch 64、lr 2e-4、40 epochs、warmup 500 和最终 checkpoint 选择均保持一致。

- `jsd`：保留原仓库实现。其实际代价为 0.5 KL(m || p) + 0.5 KL(m || qε)，
  m=(p+qε)/2；q 使用原代码 1e-12 平滑。这不是标准 JS，并且没有 log(2) 上界。
- `jsd_standard`：标准 0.5 KL(p || m) + 0.5 KL(q || m)，m=(p+q)/2。
  采用自然对数；不对 one-hot 目标平滑，0 log 0 按 0 处理。
  数值与 SciPy JS distance 的平方交叉核对，并验证梯度及极端 logits。

结果均为百分比；PER 越低越好，其余越高越好。分段采用原评估器协议：
音频独立推理、argmax 类别变化含 blank、无时间偏移、20ms 一对一边界匹配。
Overlap 为正确识别音素区间交集 / 全部参考时长；它与论文 starting-frame F1/IDR
不等价。补充 benchmark 只用于测试。CTC 的学习率沿用其 README 5e-5。

| 模型 | 测试集 | raw PER | no-sil PER | boundary F1@20ms | R-value | interval overlap |
|---|---|---:|---:|---:|---:|---:|
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

- CTC：训练 12.94 分钟，Trainer eval_wer 8.11%；checkpoint `/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ctc_fuse_phoneme/checkpoint-2920`。
- OTTC CE：训练 13.87 分钟，Trainer eval_wer 8.65%；checkpoint `/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/checkpoint-2920`。
- OTTC upstream jsd：训练 13.85 分钟，Trainer eval_wer 8.61%；checkpoint `/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme_jsd/checkpoint-2920`。
- OTTC standard JS：训练 14.27 分钟，Trainer eval_wer 15.53%；checkpoint `/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme_jsd_standard/checkpoint-2920`。

## 解读

在这次 seed 42 的实验中，原代码 `jsd` 在 README TIMIT 240 条上保持与 CE
相同的 raw PER（8.61%），边界 F1 提升 2.94 个百分点，区间 overlap 提升
2.40 个百分点；其边界 F1 在四个评估集上均高于 CE。
标准 JS 在相同超参数下的 TIMIT PER 为 15.53%，F1 为 61.53%，区间 overlap
为 70.14%，没有优于 CE。它收敛更慢；本实验没有调整学习率、loss 缩放或
训练时长，不能据此判断标准 JS 调参后的最佳性能。以上为单 seed 对比，
没有多 seed 置信区间；分段指标也不等同于论文 starting-frame F1/IDR。

统一 benchmark 图、收敛曲线和三个同语音分段对照图（PNG/PDF）位于
`reproduction/results/js_comparison/`。

Trainer 的 eval_wer 是批量 padding 下的音素编辑率，与上述单音频 raw PER 分开保存。
完整 metrics、逐音频 JSONL、后验图与 checkpoint 加载验证在各自 `reproduction/results/runexp_*` 目录。
