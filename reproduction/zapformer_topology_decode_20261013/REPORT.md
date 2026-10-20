四组 Zapformer / Libri100：固定 epoch50 checkpoint 的 S3T2** 解码复评

复用各组 step5300 的 model 与 model_avg；未训练、未改原 checkpoint 或训练历史。
音频推理不输入参考音素或时间戳。图直接使用冻结 ifmdd_losses.py 的原始音素 topology，构建自由音素循环。
k2.intersect_dense 使用精确前后向 output pruning 后 shortest_path，不使用近似 search beam；无语言模型、无音素插入惩罚。
每条最佳路径核对原始图的 src/dst 连续性、每帧覆盖、入口 S0、末尾返回音素边界以及 blank 仅在边界。
同一 BF16 声学推理的 greedy-completion 预测逐句重放历史记录，全部一致才接受对比。
评分保持同一 speech-only 参考、IPA→English-collapsed ARPAbet/TIMIT39 映射及 jiwer S/D/I，PER=100×Σ(S+D+I)/Σ参考音素数。
这是 TIMIT dev400/core-test192 监控评测，不能当作 LibriSpeech WER。test 不用于选择 checkpoint 或解码超参数。

实际解码图与参考对照：

- [b 分支 FSA](fsa_b_exact.svg)：图中红色入口输出音素，自环与其他边输出 eps。
- [完整104音素 FSA](fsa_full_exact.svg)，[实际图文本](phone_loop.fst.txt)。
- [开头120ms实际路径](fsa_actual_b_path.svg)。
- [逐句参考/解码/编辑对齐](EXAMPLES.md)，[Bricks句时间对照](si1279_reference_decode.png)，[Fat句时间对照](si1909_reference_decode.png)。
- IFMDD_GRAPH_EQUIVALENCE_PASS.json 记录104分支与当前IF-MDD原始 _add_phone_arcs 的逐边一致性；SELF_LOOP_COLLAPSE_AUDIT.json 记录全部4736句独立折叠与FSA输出一致。
- b0 b0 b1 b1 b2 b2 只输出一个 b；b0 b2 b0 b2 输出两个 b。这些结果中的重复是完整音素路径再次进入，不能通过把所有同音素合并来冒称原始解码结果改善。

final

| arm | split | greedy PER% | topology PER% | insertion% | deletion% | substitution% | argmax改动帧% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jsd_fs | development | 153.17 | 251.92 | 230.57 | 0.01 | 21.34 | 20.63 |
| jsd_fs | test | 150.05 | 247.78 | 225.26 | 0.00 | 22.52 | 20.79 |
| jsd_only | development | 94.63 | 127.22 | 98.85 | 0.53 | 27.84 | 60.25 |
| jsd_only | test | 95.18 | 129.13 | 98.16 | 0.58 | 30.38 | 61.41 |
| ce_fs | development | 109.70 | 201.70 | 180.28 | 0.03 | 21.38 | 21.56 |
| ce_fs | test | 106.42 | 198.31 | 175.71 | 0.02 | 22.59 | 22.17 |
| ce_only | development | 91.81 | 191.13 | 162.52 | 0.09 | 28.52 | 66.79 |
| ce_only | test | 92.63 | 188.05 | 157.74 | 0.07 | 30.25 | 66.94 |

cumulative_average

| arm | split | greedy PER% | topology PER% | insertion% | deletion% | substitution% | argmax改动帧% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jsd_fs | development | 112.20 | 254.43 | 220.93 | 0.12 | 33.39 | 32.09 |
| jsd_fs | test | 109.39 | 253.67 | 218.42 | 0.07 | 35.19 | 32.45 |
| jsd_only | development | 82.08 | 256.72 | 203.40 | 0.08 | 53.24 | 57.57 |
| jsd_only | test | 81.73 | 252.55 | 198.23 | 0.12 | 54.20 | 57.45 |
| ce_fs | development | 84.11 | 135.04 | 84.40 | 0.51 | 50.12 | 28.97 |
| ce_fs | test | 85.12 | 132.55 | 80.60 | 0.60 | 51.35 | 29.03 |
| ce_only | development | 83.14 | 181.07 | 120.69 | 0.18 | 60.19 | 41.70 |
| ce_only | test | 83.12 | 177.06 | 116.25 | 0.22 | 60.59 | 41.41 |

Text-free topology 路径边界指标（与 supplied-phone FA 分开）

沿用冻结评分代码的 10ms grid、包含 blank 的音素变化边界、排除同音素子状态变化及原边界匹配；R-value 使用冻结 IF-MDD 公式。
这里不复评 FA；旧 FA 图约束指标不与本表混用。边界均为全套 dev/test，未计算 equal-phone-count TSE 子集。

| arm | checkpoint | split | P@20ms% | Recall@20ms% | F1@20ms% | R-value@20ms% |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| jsd_fs | final | development | 38.02 | 49.49 | 43.00 | 42.06 |
| jsd_fs | final | test | 38.85 | 50.48 | 43.91 | 42.98 |
| jsd_fs | cumulative_average | development | 30.84 | 77.81 | 44.17 | -38.67 |
| jsd_fs | cumulative_average | test | 31.01 | 78.50 | 44.46 | -39.07 |
| jsd_only | final | development | 41.19 | 50.98 | 45.57 | 47.03 |
| jsd_only | final | test | 40.24 | 49.96 | 44.58 | 45.98 |
| jsd_only | cumulative_average | development | 25.97 | 85.01 | 39.79 | -99.57 |
| jsd_only | cumulative_average | test | 26.38 | 84.64 | 40.22 | -94.19 |
| ce_fs | final | development | 43.06 | 58.10 | 49.46 | 45.56 |
| ce_fs | final | test | 44.24 | 59.54 | 50.76 | 46.86 |
| ce_fs | cumulative_average | development | 28.82 | 85.10 | 43.05 | -72.29 |
| ce_fs | cumulative_average | test | 29.12 | 84.83 | 43.36 | -68.92 |
| ce_only | final | development | 35.33 | 55.04 | 43.03 | 28.54 |
| ce_only | final | test | 35.42 | 55.37 | 43.20 | 28.38 |
| ce_only | cumulative_average | development | 28.03 | 83.39 | 41.96 | -74.77 |
| ce_only | cumulative_average | test | 28.46 | 82.84 | 42.37 | -69.55 |

每句 raw argmax、合法路径与两种预测在各 arm 的 *_predictions.jsonl；原始计数在 results.json。
图构造、声学推理、映射、评分代码和 dev/test 数据 SHA256 在 protocol.json；历史IF-MDD raw-PHN与R-value源码hash另见scorer_provenance.json；checkpoint SHA256 在各 arm/COMPLETE.json。
新结果属于 checkpoint 固定的 evaluation，保留 epoch50/step5300；没有把重评时间当训练时间，也没有插值。
