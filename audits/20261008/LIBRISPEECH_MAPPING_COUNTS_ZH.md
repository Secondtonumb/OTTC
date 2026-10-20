# IF-MDD LibriSpeech 音素映射数量审计

统计实际保存的 monophone perceived_train_target，每次出现计一个 token；包含 sil。另存 no-sil 比例、对齐区间数量及时长。
来源为旧 IF-MDD 的 gilkeyio/librispeech-alignments 导出数据，不是 TIMIT 原始人工 PHN 标注。不存在的闭塞标签不代表录音中不存在闭塞。
原始 OTTC README 的 LibriSpeech 实验是文本字符识别；此处是用户之前 IF-MDD 的音素实验数据。

| 数据集 | 语音数 | 目标 token 总数（含 sil） | ao 个数（比例） | zh 个数（比例） | ao+zh 比例 | q/dx/六类闭塞标签 |
|---|---:|---:|---:|---:|---:|---:|
| libri100_train | 28,538 | 3,587,191 | 49,714 (1.39%) | 1,615 (0.05%) | 1.43% | 0 |
| libri960_train | 281,191 | 33,859,099 | 488,031 (1.44%) | 14,928 (0.04%) | 1.49% | 0 |
| test_clean | 2,620 | 194,742 | 2,663 (1.37%) | 92 (0.05%) | 1.41% | 0 |
| test_other | 2,938 | 187,618 | 2,570 (1.37%) | 66 (0.04%) | 1.40% | 0 |
| test_combined | 5,558 | 382,360 | 5,233 (1.37%) | 158 (0.04%) | 1.41% | 0 |

## 时长过滤与比例解释

- libri100_train：ao+zh 的 no-sil 比例 1.45%；若过滤 duration >20s 的语音，移除 1 条，ao+zh 的含 sil 比例为 1.43%。
- libri960_train：ao+zh 的 no-sil 比例 1.51%；若过滤 duration >20s 的语音，移除 52 条，ao+zh 的含 sil 比例为 1.49%。

训练 manifest 的 canonical_aligned 与 perceived_train_target 一致性已逐条统计；详细 JSON 提供全部音素及各训练子集。这些频次不能直接换算成 PER 或 segmentation F1 的变化。
