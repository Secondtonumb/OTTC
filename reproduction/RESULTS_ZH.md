# 官方 README：TIMIT fused-phoneme 复现结果

CTC 和 OTTC 都完成 40 epochs，使用官方 Wav2Vec2-large-LV60 底座和
README 超参数，未经自有 checkpoint 初始化。训练数据包含 SA，train 4,620，
kaldi_test 240；音素映射保持官方代码（含 q 的 40 类，加 pad/unk 为 42）。
最终 epoch-40 checkpoint 用于全部结果，没有按测试指标挑选 checkpoint。

下表均为百分比；PER 越低越好，边界指标与 overlap 越高越好。
识别和分段推理只提供音频。Boundary 指标采用 20ms 容忍、无拟合时间偏移、
一对一匹配；预测边界为 argmax 类别变化（包括 blank），参考为标注区间
全部端点并排除语音两端 20ms。Overlap 分子来自编辑对齐中正确识别音素
的区间交集，分母包括全部参考音素的时长。这些是明确协议下的补充分段
评估，不能冒称复现论文未完整公开的 starting-frame F1/IDR 数字。

| 模型 | 测试集 | 语音数 | raw PER | normalized no-sil PER | boundary P | boundary R | boundary F1 | R-value | overlap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OTTC | readme_kaldi_test_240 | 240 | 8.61 | 8.94 | 61.65 | 63.07 | 62.35 | 67.63 | 71.28 |
| OTTC | timit_core_test_192 | 192 | 12.07 | 9.68 | 62.13 | 63.46 | 62.79 | 68.02 | 70.33 |
| OTTC | buckeye_test | 1840 | 55.42 | 28.66 | 47.22 | 57.76 | 51.96 | 53.28 | 45.19 |
| OTTC | l2arctic_test | 900 | 41.97 | 24.31 | 57.14 | 66.29 | 61.38 | 63.77 | 54.17 |
| CTC | readme_kaldi_test_240 | 240 | 8.01 | 8.16 | 42.14 | 74.85 | 53.93 | 22.87 | 25.95 |
| CTC | timit_core_test_192 | 192 | 11.13 | 8.67 | 42.51 | 75.19 | 54.31 | 23.68 | 25.94 |
| CTC | buckeye_test | 1840 | 52.16 | 26.98 | 34.57 | 73.41 | 47.01 | -6.85 | 17.44 |
| CTC | l2arctic_test | 900 | 40.85 | 23.38 | 35.34 | 72.65 | 47.55 | -1.51 | 18.55 |

- OTTC checkpoint：`/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ottc_fuse_phoneme/checkpoint-2920`，global_step=2920。
  官方 Trainer eval_wer=8.65%；单音频 raw PER 单独列出，避免 padding 解码差异。
  训练实测 13.87 分钟（Trainer runtime，含逐 epoch 评估/保存）。
- CTC checkpoint：`/work/gm64/m64000/OTTC/reproduction/models/large_models_results/wav2vec2-large-lv60/runexp_ctc_fuse_phoneme/checkpoint-2920`，global_step=2920。
  官方 Trainer eval_wer=8.11%；单音频 raw PER 单独列出，避免 padding 解码差异。
  训练实测 12.94 分钟（Trainer runtime，含逐 epoch 评估/保存）。

环境、必要兼容修复、启动命令与 metric contract 见 `reproduction/README.md`；底座来源见 `reproduction/provenance/upstream.json`；每组 `results/runexp_*_fuse_phoneme/` 内有 runtime/source hashes、完整逐语音 JSONL、10/20/30ms 指标、编辑计数和完成标记。
