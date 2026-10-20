# 最早 TIMIT 单头 OTTC：K2-FSA forced alignment

本次补测原始 CE / upstream JSD 的 epoch 40、step 2920 checkpoint。模型为 wav2vec2-large-lv60，原生输出间隔 20 ms，单头 42 类；没有 S3T2。

同一套 TIMIT core-test 192 句。FA 给定历史 manifest 的参考音素顺序（包含 sil），不向模型或对齐器输入参考时间戳；native K2 CTC graph → DenseFsaVec → pruned intersection → shortest_path，没有替代后端。

| OTTC 代价 | Text-free F1@20ms | Text-free R-value | K2 FA F1@20ms | K2 FA R-value | FA 边界 MAE |
|---|---:|---:|---:|---:|---:|
| CE | 62.79% | 68.02% | 59.13% | 65.28% | 22.44 ms |
| 原上游 JSD | 65.36% | 70.25% | 61.46% | 67.24% | 21.13 ms |

JSD 相对 CE：FA F1 +2.33 个百分点；R-value +1.97 个百分点；边界 MAE -1.32 ms。

| OTTC 代价 | FA P@20ms | FA Recall@20ms | start MAE | end MAE | start+end TSE | 对齐失败 |
|---|---:|---:|---:|---:|---:|---:|
| CE | 59.66% | 58.61% | 22.01 ms | 22.88 ms | 44.89 ms | 0 / 192 |
| 原上游 JSD | 62.01% | 60.92% | 20.70 ms | 21.55 ms | 42.26 ms | 0 / 192 |

评分与解释：

- 边界 P/R/F1：所有参考 interval 的唯一端点，去掉距句首/句尾不超过 20 ms 的端点；一对一、仅时间匹配，容差 10/20/30 ms。R-value 使用冻结 IF-MDD 公式，源码哈希保存在 metrics.json。
- FA 时间戳使用 IF-MDD 的 duration / frame_count 网格，blank 归前一个音素；旧 text-free 使用 frame_index × 0.02。未调时间偏移。
- FA 边界 MAE 是全部给定音素的 start/end 绝对误差平均；start+end TSE 是每音素两个误差的和，因此为 MAE 的两倍。没有筛选正确识别的音素。
- 历史 core192 标注已 dx→d、去 q；去 q 留下的参考时间空隙保留不填补，评分仍保留空隙两端。未更改音素标签或移除 sil。
- 全部 192 句的旧预测音素、分割区间和边界已逐句精确回放；CE/JSD 对齐路径均逐句验证折叠后完全等于给定音素顺序。
- FA 的音素序列已知，因此不能用其 PER 表示识别能力。这是 supplied-phone FA，不是词文本 G2P 输入的 FA-Benchmark，也不能直接等同于 raw-PHN MFA topline。

结果文件：`full/ce/metrics.json`、`full/jsd_upstream/metrics.json`；逐句路径和区间：对应目录 `predictions.jsonl`。W&B 的独立 evaluation run 在对应 `wandb_run.json`。历史训练及旧评分文件保持原样。
