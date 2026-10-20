# Buckeye 单头 OTTC：CE / 原上游 JSD / standard JSD

状态：三组最终评测完成。

已有原始人工标注19位说话人子集：train 10,927句（15位，7.33小时）、dev 1,524句（2位，1.11小时）、test 1,840句（2位，0.87小时）。说话人、句子、源录音互斥；不是完整40位Buckeye。

三组保留同一自监督预训练wav2vec2-large-lv60，随机phone/mass单头。CNN冻结、Transformer微调。20ms、seed42、batch64、AdamW lr2e-4、weight decay0.005、warmup500、40epoch/6840step、regfrozen30epoch，BF16 autocast与FP32主权重。固定最终epoch40评测，不用test选择checkpoint。

只改变OTTC运输代价：CE；原上游JSD（0.5 KL(m||p)+0.5 KL(m||q)，q按上游加epsilon）；standard JSD（0.5 KL(p||m)+0.5 KL(q||m)）。m=(p+q)/2。standard复用已有jsd_standard实现。CE在OTTC内部，无帧级CE。三种raw loss尺度不同，不直接比较数值高低。

训练仅输入音频与音素序列，沿用perceived_train_target及边缘sil、合并相邻重复标签，ao→aa。最终评分保留所有人工speech-phone区间，不合并参考时间区间。最终PER来自同一FP32独立评测器；训练日志的BF16诊断PER采用折叠训练目标，不能混用。

K2 FA输入真实speech-phone顺序，无参考时间戳、无人工edge-sil。native K2 CTC graph、DenseFsaVec、pruned intersection、shortest_path，blank归前一音素，时间网格duration/frames，无偏移。唯一区间端点、一对一时间匹配、去句首/句尾20ms，R-value使用冻结IF-MDD公式。

| Split | OTTC代价 | K2 FA F1@20ms ↑ | K2 FA R-value ↑ | FA边界MAE ↓ | 对齐失败 |
|---|---|---:|---:|---:|---:|
| dev | CE | 53.81% | 60.80% | 27.68 ms | 0 / 1524 |
| dev | 原上游 JSD | 54.26% | 61.18% | 27.35 ms | 0 / 1524 |
| dev | standard JSD | 55.68% | 62.38% | 26.77 ms | 0 / 1524 |
| test | CE | 49.45% | 57.15% | 29.82 ms | 0 / 1840 |
| test | 原上游 JSD | 49.93% | 57.55% | 29.24 ms | 0 / 1840 |
| test | standard JSD | 50.55% | 58.07% | 29.69 ms | 0 / 1840 |

MAE使用当前已完成3组共同成功对齐句子：dev 1524句、test 1840句；三组完成后统一用三组交集。F1/R保留完整集合，失败句记零命中。

| Split | OTTC代价 | Text-free 去sil PER ↓ | Text-free F1@20ms ↑ | Text-free R-value ↑ |
|---|---|---:|---:|---:|
| dev | CE | 16.24% | 56.58% | 62.34% |
| dev | 原上游 JSD | 15.48% | 58.58% | 63.83% |
| dev | standard JSD | 17.08% | 58.53% | 63.83% |
| test | CE | 20.19% | 52.00% | 57.47% |
| test | 原上游 JSD | 19.64% | 54.06% | 58.97% |
| test | standard JSD | 21.15% | 53.67% | 58.78% |

Test 原上游 JSD相对CE：PER -0.55个百分点；K2 FA F1 +0.48个百分点，R-value +0.40个百分点。

Test standard JSD相对CE：PER +0.96个百分点；K2 FA F1 +1.10个百分点，R-value +0.92个百分点。

Test standard JSD相对原上游JSD：PER +1.52个百分点；K2 FA F1 +0.62个百分点，R-value +0.52个百分点。

FA音素标签已知，不报告FA PER。这里是给定音素FA，与参考词/G2P输入的FA-Benchmark分开。此处仅一个seed的同配方比较。

每组完整指标与逐句路径位于evaluation/{ce,jsd_upstream,jsd_standard}。训练及评测独立记录W&B，身份保存在对应目录的wandb_run.json。
