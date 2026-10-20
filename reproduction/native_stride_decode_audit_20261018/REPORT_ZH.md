# TIMIT 原生10ms：K2 S3T2** 重新解码

更新 UTC：2026-10-18T08:48:29.734624+00:00。dev400/core-test192。原权重不变，无新训练。

本次独立FP32模型解码，不沿用Trainer训练时的BF16 autocast包装。旧评测的“FP32”标签不准确；旧数值保留在原目录，本报告采用新结果。

解码直接复用IF-MDD原始token FST：S0与S2必经、S1可跳过、状态允许self-loop，sil单状态，保留重复音素；K2 output_beam=8，无phone LM、插入惩罚或给定转录。

在既定best-dev/final/final5/final10四个候选中，仅用dev K2 PER选择。训练期best-dev候选本身按greedy选出；本报告不将其称为所有epoch中的K2最优。

无文本core-test指标（PER及P/R/F1/R-value均为%；边界统一固定10ms时间轴、20ms容差）：

| 输出/解码 | OTTC代价 | dev所选权重 | dev PER | test PER | P | R | F1 | R-value |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 单音素 greedy | jsd_upstream | final_10 | 13.42 | 14.75 | 59.42 | 57.24 | 58.31 | 64.72 |
| 单音素 greedy | ce | best_dev | 18.08 | 19.90 | 59.38 | 55.59 | 57.42 | 64.12 |
| 单音素 greedy | jsd_standard | final_10 | 14.99 | 16.27 | 55.51 | 49.97 | 52.59 | 60.33 |
| S3T2** K2 | jsd_upstream | final | 12.96 | 14.77 | 52.87 | 49.31 | 51.03 | 58.89 |
| S3T2** K2 | ce | final_10 | 15.42 | 16.88 | 51.79 | 44.99 | 48.15 | 56.91 |
| S3T2** K2 | jsd_standard | final | 12.00 | 13.38 | 51.22 | 42.82 | 46.64 | 55.84 |

S3T2** 原 duration/frame_count 时间轴的无文本分割分数（与原解码导出兼容，不能与固定10ms时间轴混用）：

| 代价 | F1@20 | R-value@20 |
|---|---:|---:|
| jsd_upstream | 48.23 | 56.60 |
| ce | 46.34 | 55.49 |
| jsd_standard | 45.87 | 55.24 |

给定音素FA单独报告（非无文本分割；时间轴duration/frame_count）：

| 输出 | 代价 | P | R | F1 | R-value | MAE ms | 失败/core192 |
|---|---|---:|---:|---:|---:|---:|---:|
| mono | jsd_upstream | 56.57 | 55.57 | 56.07 | 62.69 | 27.79 | 0/192 |
| mono | ce | 55.19 | 54.22 | 54.70 | 61.53 | 29.12 | 0/192 |
| mono | jsd_standard | 50.55 | 49.66 | 50.10 | 57.64 | 31.52 | 0/192 |
| s3t2ss | jsd_upstream | 50.05 | 49.17 | 49.60 | 57.22 | 43.40 | 0/192 |
| s3t2ss | ce | 49.10 | 48.23 | 48.66 | 56.42 | 42.50 | 0/192 |
| s3t2ss | jsd_standard | 48.68 | 47.82 | 48.25 | 56.07 | 45.83 | 0/192 |

验证：SAVED_PATH_AUDIT.json保存原7104条无文本路径审查；CORRECTED_PATH_AUDIT.json审查重新解码的7104条无文本路径和7104条FA路径，全部符合原始拓扑且覆盖完整。

修复边界：在独立新评测入口中，K2图构建放到inference_mode(False)，模型从普通FP32参数重建。原训练源、epoch权重、平均权重和历史结果均保留。旧FA全失败对应分数无效。

Libri100仍未提交，8卡只用于未来经结果审查放行后的Libri100。WavLM已完成训练的九组评测错误另在evaluation_dispatch.json记录恢复作业，不重训。
